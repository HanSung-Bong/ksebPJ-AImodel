def save_to_textfile(data_list,filename):
  with open(filename,'w') as file:
    for sublist in data_list:
      line = ' '.join(map(str,sublist))
      file.write(line + '\n')