import tkinter as tk
from tkinter import filedialog, messagebox
from model.data_preprocessing import preprocess_data
from model.model_training import train_model, evaluate_model

class SportsPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sports Game Outcome Prediction")

        self.label = tk.Label(root, text="Select CSV File:")
        self.label.pack()

        self.button = tk.Button(root, text="Browse", command=self.load_file)
        self.button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                X_train, X_test, y_train, y_test = preprocess_data(file_path)
                model = train_model(X_train, y_train)
                accuracy = evaluate_model(model, X_test, y_test)
                self.result_label.config(text=f"Accuracy: {accuracy:.2f}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SportsPredictionApp(root)
    root.mainloop()
