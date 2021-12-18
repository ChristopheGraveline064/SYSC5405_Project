from sklearn.metrics import mean_squared_error

dir = "./MSE/"
#mine = "Group10_blind_predictions.txt"
mine = "Group10_blind_predictions_3.txt"
#his = "Full_train_blind_test_prob.txt"
his = "Group10.txt"

my_path = dir + mine
his_path = dir + his
n = 4
my_file = open(my_path)
my_file_contents = my_file.read()
my_contents_split = my_file_contents.splitlines()
my_result = [float(value) for value in my_contents_split]
print(my_contents_split[n])
print(my_result[n])
my_file.close()

his_file = open(his_path)
his_file_contents = his_file.read()
his_contents_split = his_file_contents.splitlines()
#print(his_contents_split)
his_result = [float(value) for value in his_contents_split]
print(his_contents_split[n])
print(his_result[n])
his_file.close()

mse = mean_squared_error(my_result, his_result)
print("mse = {}".format(mse))