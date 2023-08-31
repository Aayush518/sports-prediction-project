import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
import pickle
from model.data_preprocessing import preprocess_data
from model.model_training import train_model, evaluate_model

class SportsPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sports Game Outcome Prediction")
        self.title_label = ttk.Label(root, text="Sports Game Outcome Prediction", font=("Helvetica", 18, "bold"))
        self.title_label.pack(pady=20)
        self.label = ttk.Label(root, text="Select CSV File:", font=("Helvetica", 12))
        self.label.pack()
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(root, textvariable=self.file_path_var, font=("Helvetica", 10), state="readonly")
        self.file_entry.pack(pady=5, padx=20, fill="x")
        self.browse_button = ttk.Button(root, text="Browse", command=self.load_file)
        self.browse_button.pack(pady=5)
        self.result_label = ttk.Label(root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_var.set(file_path)
            X_train, X_test, y_train, y_test = preprocess_data(file_path)
            model = train_model(X_train, y_train)
            accuracy = evaluate_model(model, X_test, y_test)
            self.result_label.config(text=f"Accuracy: {accuracy:.2f}%")

            progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
            progress_bar.pack(pady=10)
            progress_bar["maximum"] = 100

            for i in range(0, 101, 20):
                progress_bar["value"] = i
                root.update_idletasks()
                time.sleep(1)

            save_button = ttk.Button(root, text="Save Model", command=lambda: self.save_model(model))
            save_button.pack(pady=5)

    def save_model(self, model):
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)
        messagebox.showinfo("Success", "Model saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SportsPredictionApp(root)
    root.mainloop()
