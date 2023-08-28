import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from ttkthemes import ThemedStyle
from model.data_preprocessing import preprocess_data
from model.model_training import train_model, evaluate_model

class SportsPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sports Game Outcome Prediction")
        self.style = ThemedStyle(root)
        self.style.set_theme("clam")  # Choose a theme (e.g., 'arc', 'radiance', 'adapta')

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
        # Display a file dialog to choose a CSV file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_var.set(file_path)
            try:
                # Preprocess the data
                X_train, X_test, y_train, y_test = preprocess_data(file_path)

                # Train the model
                model = train_model(X_train, y_train)

                # Evaluate the model and display accuracy
                accuracy = evaluate_model(model, X_test, y_test)
                self.result_label.config(text=f"Accuracy: {accuracy:.2f}%")

            except Exception as e:
                # Display an error message if an exception occurs
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SportsPredictionApp(root)
    root.mainloop()
