A2au = 1.8897261246  # 1 Å = 1.8897 a.u.
au2A = 0.52917721091
eV2au = 0.036749308136648
au2eV = 27.211396641308  # 1 a.u. = 27.211 eV


def split_str2numtext(text):
    """
    Splits provided string into tuple of text and numbers

    Parameters
    ----------
    text : An input string for splitting.

    Returns
    -------
    tuple

    """
    result = []
    temp = text[0]
    prev_is_num = text[0].isdigit()
    for i, s in enumerate(text[1:]):
        if s.isdigit() and prev_is_num:
            temp += s
        elif not s.isdigit() and prev_is_num:
            result.append(int(temp))
            temp = ''
            prev_is_num = False
        if not s.isdigit() and not prev_is_num:
            temp += s
        elif s.isdigit() and not prev_is_num:
            result.append(temp)
            temp = s
            prev_is_num = True
    try:
        temp = int(temp)
    except TypeError:
        pass
    result.append(temp)
    return tuple(result)


def iseven(val):
    """
    Checks if number is even.

    Parameters
    ----------
    val : int
        Number to check.

    Returns
    -------
    bool
        True if even.

    """
    if val % 2 == 0:
        return True
    else:
        return False
