import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db
import google.generativeai as genai
import smtplib
import datetime
from dotenv import load_dotenv
import os
import json
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Initialize Firebase
cred = credentials.Certificate("C:\SavEat\saveat-adfcc-firebase-adminsdk-fbsvc-84e935c326.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://saveat-adfcc-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Load spoilage detection model only
spoilage_model = load_model("models/fruit_spoil_detector.h5")

# Class labels for spoilage prediction
spoilage_labels = ["freshapples", "freshbanana", "freshcucumber", "freshokra", "freshoranges", "freshpotato", "freshtomato", "rottenapples", "rottenbanana", "rottencucumber", "rottenokra", "rottenoranges", "rottenpotato", "rottentomato"]

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyClbNDVX0kvHstzS1Bs_ribaFhH9TQZSpg"))

# Email credentials
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("USER_EMAIL")

# Twilio credentials
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_TO = os.getenv("USER_PHONE_NUMBER")

def capture_image():
    """Capture an image from the laptop camera"""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Scan Food Item", frame)
        if cv2.waitKey(1) & 0xFF == ord("s"):  # Press 's' to capture
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def preprocess_image(img):
    """Resize and normalize image for model input"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (128, 128))  # Ensure correct input size
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

def predict_food_status(img):
    """Predicts if the food is fresh or spoiled"""
    processed_img = preprocess_image(img)
    prediction = spoilage_model.predict(processed_img)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    CONFIDENCE_THRESHOLD = 0.6 

    # Ensure index is within bounds
    if predicted_index < len(spoilage_labels):
        result = spoilage_labels[predicted_index]
    
    elif confidence < CONFIDENCE_THRESHOLD:
        result = "Invalid"  

    else:
        result = "Unknown"

    print("Model predicted class probabilities:", prediction)
    print("Predicted class index:", predicted_index)
    print("Available class labels:", spoilage_labels)
    print("Total classes:", len(spoilage_labels))

    return result


def list_available_models():
    models = genai.list_models()
    for model in models:
        print(model.name)

list_available_models()

def scan_food():
    """Scans and processes a food item"""
    print("Scanning food item...")
    img = capture_image()
    food_status = predict_food_status(img)
    
    if "rotten" in food_status:
        print(f"{food_status} - Spoiled!")
        food_name = food_status.replace("rotten", "").strip()
        recipe = get_recipe(food_name)
        send_sms_alert(food_name)
    else:
        print(f"{food_status} - Fresh!")
    
def get_recipe(food_name):
    """Generates a recipe suggestion using Gemini API."""
    prompt = f"Suggest an easy recipe using {food_name} before it spoils."

    # Ensure the Gemini API is correctly configured
    genai.configure(api_key=os.getenv("AIzaSyClbNDVX0kvHstzS1Bs_ribaFhH9TQZSpg"))

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        if response and response.text:
            recipe = response.text.strip()
            print(f"🔹 Recipe Suggestion:\n{recipe}")
            return recipe
        else:
            print("⚠️ Gemini API did not return a valid response.")
            return "No recipe suggestion available."
    
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Failed to generate a recipe."

def send_email_alert(food_name, recipe):
    """Sends an email alert with the Gemini-generated recipe."""
    sender_email = os.getenv("EMAIL")
    sender_password = os.getenv("EMAIL_PASSWORD")
    receiver_email = os.getenv("USER_EMAIL")

    subject = f"⚠️ Alert: {food_name.title()} May Spoil Soon!"
    body = f"""
    Hello Achiever 👋,

    Your {food_name} may spoil soon. Here's a quick recipe you can try before it's too late:

    🍽️ Recipe Suggestion:
    {recipe}

    SavEat – Smart way to reduce grocery waste! 🧠🥕
    """

    try:
        # Create email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        # Connect and send
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("📧 Email alert sent successfully!")

    except Exception as e:
        print(f"❌ Email sending failed: {e}")

def send_sms_alert(food_name):
    """Generates a recipe, prints it, and sends it as SMS via Twilio"""
    recipe = get_recipe(food_name) 

    if recipe.startswith("Failed") or recipe == "No recipe suggestion available.":
        print("⚠️ No valid recipe generated. Skipping SMS alert.")
        return
    
    body = f"⚠️ Alert: Your {food_name} may spoil soon!\nTry this quick recipe:\n\n{recipe}"

    # Initialize Twilio client
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"✅ Alert sent successfully. Message SID: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS alert: {e}")

    # ✅ Output the generated recipe
    print("\n🔹 Final Recipe Output:\n", recipe)


def add_expiry_to_firebase(food_name, expiry_date):
    """Stores packaged food expiry date in Firebase"""
    ref = db.reference("packaged_food")
    ref.child(food_name).set({"expiry_date": expiry_date})
    print(f"Added {food_name} expiry date to Firebase: {expiry_date}")

def check_expiry_dates():
    """Checks Firebase for soon-to-expire items and sends alerts"""
    ref = db.reference("packaged_food")
    data = ref.get()
    if data:
        today = datetime.date.today()
        for food, details in data.items():
            expiry_date = datetime.datetime.strptime(details["expiry_date"], "%Y-%m-%d").date()
            if (expiry_date - today).days == 5:
                send_sms_alert(food)

def main():
    while True:
        print("\n1. Scan Fruits/Vegetables")
        print("2. Add Packaged Food Expiry")
        print("3. Check Expiry Alerts")
        print("4. Exit")
        choice = input("Enter choice: ")
        
        if choice == "1":
            scan_food()
        elif choice == "2":
            food_name = input("Enter packaged food name: ")
            expiry_date = input("Enter expiry date (YYYY-MM-DD): ")
            add_expiry_to_firebase(food_name, expiry_date)
        elif choice == "3":
            check_expiry_dates()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
