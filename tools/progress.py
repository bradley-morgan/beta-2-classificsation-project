from tqdm import tqdm


def make_progress_bar(total, description='', position=0):
    return tqdm(total=total, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                desc=description, position=position)
