def prob_to_sample(probs, batch_size, num_steps):
    outputs = []
    for i in range(batch_size):
        seq = []
        for j in range(num_steps):
            char = np.random.multinomial(1, probs[i, j]/(np.sum(probs[i,j])+1e-5))
            char = np.argmax(char)
            seq.append(char)
        outputs.append(num_to_string(seq))
    return outputs

def num_to_string(num_list):
    string_list = []
    char_dict = {0:'A',1:'G',2:'C',3:'T',4:' ',5:'|',6:' ',7:' '}
    for num in num_list:
        if num == 6:
            break
        char = char_dict[num]
        string_list.append(char)
    if not string_list:
        string_list.append(' ')
    return ''.join(string_list)