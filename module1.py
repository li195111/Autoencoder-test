import os
model_dir = os.path.join(os.getcwd(), "model")
files = []
for file in os.listdir(model_dir):
	files.append(file.split('.')[0])
times_str = [file.replace('autoencoder_model', '') for file in files if 'autoencoder_model' in file]
times_list = []
for time_str in times_str:
	if time_str != '':
		times = int(time_str)
		times_list.append(times)
	else:
		times = 0
		times_list.append(times)
max_times = max(times_list)
print (max_times)
