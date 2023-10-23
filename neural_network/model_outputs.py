

def model_output_to_english(li: list) -> str:
    """
    given list from output of model EX tensor([[0.3765, 0.4338, 0.3295]])
    turn this back into what the hand is doing

    :param li:
    :return:
    """
    output = ''
    if li[2] < li[0] > li[1]:
        output = 'Hand down'
    if li[2] < li[1] > li[0]:
        output = 'Hand up'
    if li[0] < li[2] > li[1]:
        output = 'hand stays still'
    return output