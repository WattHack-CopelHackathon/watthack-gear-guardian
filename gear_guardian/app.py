import tkinter as tk
import threading
from PIL import ImageTk, Image
import cv2

class App(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        self.root = tk.Tk()
        self.root.title("Detecção de EPIs")
        self.root.geometry("1920x1080")
        self.root.protocol("WM_DELETE_WINDOW", self.callback)
        
        # Orange color theme
        orange_color = "#FF7000"
        text_color = "#FFFFFF"

        # Applying styles
        self.root.configure(bg=orange_color)

        label1 = tk.Label(self.root, text="Pessoas com capacete = ", bg=orange_color, fg=text_color)
        label1.place(x=840, y=750)
        
        label2 = tk.Label(self.root, text="Pessoas sem capacete = ", bg=orange_color, fg=text_color)
        label2.place(x=840, y=770)
        
        label5 = tk.Label(self.root, text="Mensagens", relief=tk.RIDGE, bg=orange_color, fg=text_color)
        label5.place(x=920, y=730)

        self.num_with_helmet1 = tk.StringVar()
        label_with_helmet1 = tk.Label(self.root, textvariable=self.num_with_helmet1, bg=orange_color, fg=text_color)
        label_with_helmet1.place(x=980, y=750)

        self.num_without_helmet1 = tk.StringVar()
        label_without_helmet1 = tk.Label(self.root, textvariable=self.num_without_helmet1, bg=orange_color, fg=text_color)
        label_without_helmet1.place(x=980, y=770)

        self.messages1 = tk.Text(self.root, bd=5, relief=tk.RIDGE)
        self.messages1.place(x=710, y=800, width=500, height=180)

        # img = cv2.imread('no_signal.jpg', 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # img = ImageTk.PhotoImage(image=img)

        # self.panel1 = tk.Label(image=img)
        # self.panel1.image = img
        # self.panel1.place(x=700, y=10)

        quit_btn = tk.Button(self.root, text="Sair", command=self.root.quit, width=10, height=2, bg=orange_color, fg=text_color, relief=tk.RIDGE)
        quit_btn.place(x=920, y=990)

        self.root.mainloop()

    def callback(self):
        self.root.quit()

    def quit(self):
        self.root.destroy()

    def update(self, images, values, messages):
        img = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.panel1.configure(image=img)
        self.panel1.image = img

        self.num_with_helmet1.set(values[0])
        self.num_without_helmet1.set(values[1])
        message = messages[0]
        if message != "":
            self.messages1.insert(tk.INSERT, message)
