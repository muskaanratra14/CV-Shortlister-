import tkinter as tk
#import requests
from temp import make_prediction
import re

HEIGHT = 500
WIDTH = 600

def popup(prediction):
    win = tk.Toplevel()
    win.geometry("300x200+300+400")
    if prediction:
        final_str='Selected'
    else:
        final_str='Rejected'
    text = tk.Label(win, text=final_str)    
    text.grid(row=4,column=2)
    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=5, column=6)

def format_response(prediction):
    if prediction:
        final_str='Selected'
    else:
        final_str='Rejected'
    
    return final_str
#
def get_testResume(filename):
    match = re.search(r'\d+', filename)
    if match:
        pdfno = int(match.group())
    label['text'] = popup(make_prediction(pdfno)) #
    print(filename)
    #print(label['text'])



root = tk.Tk()
#root.resizable(0,0)
root.title("CV Shortlister") 

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack(fill=tk.BOTH, expand=1)

photo=tk.PhotoImage(file= 'C://Users//Muskaan Ratra//Downloads//cv_PNG1.png')
canvas.create_image(0,0, image=photo)

frame = tk.Frame(root, bg='black', bd=5) #80c1ff'
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

entry = tk.Entry(frame, font=40)
entry.place(relwidth=0.65, relheight=1)

button = tk.Button(frame, text="upload", font=20, bg='red', command=lambda: get_testResume(entry.get()))
button.place(relx=0.7, relheight=1, relwidth=0.3)

lower_frame = tk.Frame(root, bg='grey', bd=10)#80c1ff
lower_frame.place()

label = tk.Label(lower_frame)
label.place(relwidth=1, relheight=1)

root.mainloop()

