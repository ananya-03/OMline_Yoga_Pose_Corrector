import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
from PIL import Image, ImageTk


# ---------- FUNCTION TO LAUNCH YOUR SCRIPT ----------
def start_pose_detection():
    if app.pose_running:
        messagebox.showinfo("Yoga Pose Buddy", "Pose detection is already running.")
        return

    def run_script():
        app.pose_running = True
        try:
            subprocess.run(["python", "yoga_pose_detector.py"])  # Your existing script name
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run script:\n{e}")
        finally:
            app.pose_running = False

    threading.Thread(target=run_script, daemon=True).start()

# ---------- GUI SETUP ----------
app = tk.Tk()
app.title("🧘‍♀️ OMLine 🧘")
# app.geometry("500x300")
app.configure(bg="#e0f7fa")
app.pose_running = False

# Load and place background image and set window size
bg_image = Image.open("Omline.png")  # Replace with your image
width, height = bg_image.size
app.geometry(f"{width}x{height}")  # ✅ Set window size to image size

bg_image = bg_image.resize((width, height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(app, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# title_label = tk.Label(app, text="Yoga Pose Buddy", font=("Segoe UI", 24, "bold"), bg="#e0f7fa", fg="#00796b")
# title_label.pack(pady=40)

# start_button = tk.Button(
#     app,
#     text="Start Pose Detection",
#     font=("Segoe UI", 16),
#     bg="#4caf50",
#     fg="black",
#     padx=20,
#     pady=10,
#     relief="raised",
#     bd=3,
#     command=start_pose_detection
# )
# start_button.place(relx=0.5, rely=0.1, anchor="center")  # Top middle

# exit_button = tk.Button(
#     app,
#     text="Exit",
#     font=("Segoe UI", 12),
#     bg="#ef5350",
#     fg="black",
#     padx=10,
#     pady=5,
#     command=app.destroy
# )
# exit_button.place(relx=0.5, rely=0.2, anchor="center")   # Slightly below start

# Define hover behavior for buttons
def on_enter(e):
    e.widget['bg'] = '#388e3c'  # Darker green on hover (for start)
    e.widget['cursor'] = 'hand2'

def on_leave(e):
    e.widget['bg'] = '#4caf50'  # Original green

def on_exit_enter(e):
    e.widget['bg'] = '#c62828'  # Darker red on hover (for exit)
    e.widget['cursor'] = 'hand2'

def on_exit_leave(e):
    e.widget['bg'] = '#ef5350'  # Original red

# Start Button
start_button = tk.Button(
    app,
    text="Start Pose Detection",
    font=("Segoe UI", 14, "bold"),
    bg="#4caf50",
    fg="black",
    activebackground="#2e7d32",
    activeforeground="white",
    padx=16,
    pady=8,
    bd=0,
    relief="flat",
    command=start_pose_detection  

)
start_button.place(relx=0.5, rely=0.1, anchor="center")
start_button.bind("<Enter>", on_enter)
start_button.bind("<Leave>", on_leave)

# Exit Button
exit_button = tk.Button(
    app,
    text="Exit",
    font=("Segoe UI", 12, "bold"),
    bg="#ef5350",
    fg="black",
    activebackground="#b71c1c",
    activeforeground="white",
    padx=12,
    pady=6,
    bd=0,
    relief="flat",
    command=app.destroy
)
exit_button.place(relx=0.5, rely=0.2, anchor="center")
exit_button.bind("<Enter>", on_exit_enter)
exit_button.bind("<Leave>", on_exit_leave)

if __name__ == "__main__":
    app.mainloop()
