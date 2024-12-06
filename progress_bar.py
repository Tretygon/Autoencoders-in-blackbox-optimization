def progress_bar(before, current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    # ending = '\n' if current == total else '\r'

    print(f'{before} [{arrow}{padding}] {round(fraction*100,2)}%', end='\r')