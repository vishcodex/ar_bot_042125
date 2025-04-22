import os
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# --- Email Configuration ---
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") # Use App Password if MFA is enabled
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.office365.com") # Default for Outlook
SMTP_PORT_STR = os.getenv("SMTP_PORT", "587") # Default for Outlook TLS

# Validate essential configuration
if not SENDER_EMAIL:
    raise ValueError("SENDER_EMAIL not found in .env file.")
if not SENDER_PASSWORD:
    raise ValueError("SENDER_PASSWORD not found in .env file.")
if not SMTP_SERVER:
    raise ValueError("SMTP_SERVER not found in .env file.")
if not SMTP_PORT_STR or not SMTP_PORT_STR.isdigit():
     raise ValueError("SMTP_PORT must be a valid number.")

SMTP_PORT = int(SMTP_PORT_STR)

# --- MCP Server Setup ---
server = FastMCP('Email Server')

@server.tool()
async def send_email_summary(recipient_email: str, summary_text: str) -> str:
    """Sends an email summary to the specified recipient."""
    if not recipient_email:
        return "Error: Recipient email address is required."
    if not summary_text:
        return "Error: Summary text cannot be empty."

    message = EmailMessage()
    message['From'] = SENDER_EMAIL
    message['To'] = recipient_email
    message['Subject'] = "Chat Summary"
    message.set_content(summary_text)

    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        print(f"Attempting to send email to {recipient_email} via {SMTP_SERVER}:{SMTP_PORT}")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp_server:
            smtp_server.starttls(context=context)  # Secure the connection
            smtp_server.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp_server.send_message(message)
        print(f"Email successfully sent to {recipient_email}")
        return f"Email summary successfully sent to {recipient_email}."
    except smtplib.SMTPAuthenticationError:
        print("Error: SMTP authentication failed. Check SENDER_EMAIL and SENDER_PASSWORD.")
        return "Error: Email authentication failed. Please check server configuration."
    except smtplib.SMTPConnectError:
        print(f"Error: Could not connect to SMTP server {SMTP_SERVER}:{SMTP_PORT}.")
        return f"Error: Could not connect to email server {SMTP_SERVER}:{SMTP_PORT}."
    except smtplib.SMTPServerDisconnected:
         print("Error: SMTP server disconnected unexpectedly.")
         return "Error: Email server disconnected unexpectedly."
    except Exception as e:
        print(f"Error sending email: {e}")
        return f"Error sending email: {str(e)}"

if __name__ == '__main__':
    print(f"Starting Email MCP Server (PID: {os.getpid()})...")
    print(f"Configured Sender: {SENDER_EMAIL}")
    print(f"Configured SMTP: {SMTP_SERVER}:{SMTP_PORT}")
    server.run()