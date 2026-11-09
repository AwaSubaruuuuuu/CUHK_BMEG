import os
import sys

sys.path.insert(0, os.getcwd())
import tkinter as tk

import main as nf

root = tk.Tk()
app = nf.App(root)
root.update_idletasks()
root.update()

app.nb.select(1)
tab = app.tab_capture
tab.set_panel("math")
root.update_idletasks()
root.update()

print("root", root.winfo_width(), root.winfo_height())
print("tab w", tab.winfo_width())
print("q_text", repr(tab.q_text.cget("text")), "box w", tab.q_text.master.winfo_width(),
      "text need w", tab.q_text.winfo_reqwidth(), "h", tab.q_text.winfo_reqheight())
print("q_tip right", tab.q_tip.winfo_rootx() - tab.winfo_rootx() + tab.q_tip.winfo_width())
print("ops right", tab.rep_entry.master.winfo_rootx() - tab.winfo_rootx()
      + tab.rep_entry.master.winfo_width())
print("log bottom", tab.log.winfo_y() + tab.log.winfo_height(), "tab h", tab.winfo_height())

tab.set_panel("rest")
root.update_idletasks()
print("rest text", repr(tab.q_text.cget("text")))
print("rest tip", repr(tab.q_tip.cget("text")))
print("entry state", str(tab.rep_entry.cget("state")))
root.destroy()
print("layout ok")
