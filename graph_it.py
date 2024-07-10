import os, webbrowser  # to show post-processed results in the browser
import cocoex, cocopp  # experimentation and post-processing modules

exs = os.listdir(os.getcwd() + "/exdata/")
exs = sorted(exs)
seen = dict()
for ex in exs:
    if ex.endswith('.pickle'):continue
    sp = ex.split('-')
    seen[sp[0]]="exdata\\" + ex
cocopp.genericsettings.isConv = True
cocopp.main(' '.join(seen.values()))
# cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")