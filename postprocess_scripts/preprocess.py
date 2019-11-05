import os
import sys
import re

def call_retrieve_for_all_files( dirname):

	#print (dirname)
        for _dir in os.listdir(dirname):
		print _dir
                x = os.path.join(dirname, _dir, "test_final_results110")
		print x
                if (os.path.exists(x)):
                        m = "python retreive.py "  +  x
                        os.system(m)


        #list_experiments = ["aaaa", "aaba", "aabb", "aaab"]
        #models = ["ds", "dh", "h", "s", "vm", "od", "sds_s", "sds", "sds_os", "sds_oh"]
	#models = ["ds_f_ct_pluse"]
	#for i in list_experiments:
	#	print i
        #       for j in models:
	#		print j
        #                temp = "results_" + i + "_" + j

        #                if not (os.path.exists(temp)):
        #                        os.makedirs(temp)



        for _dir in os.listdir(dirname):
		y = re.search("b", _dir)

		if y is None:
			continue
		y = y.start()

		if y is None:
			continue

                temp = _dir[y:]
                new_dest = "results_" + _dir

                file_dest = os.path.join(dirname, _dir, "test_final_results110_plabels")

                new_dir = os.path.join(dirname,  new_dest, _dir[y:])
		print new_dir
                if not (os.path.exists(new_dir)):
                        os.makedirs(new_dir)

                command = "cp " + file_dest + "  " + new_dir + "/test_final_results110_plabels"
         	print (command)
                os.system(command)


                #transfer_true_file = os.path.join("../diff_query_data/", _dir[y:],"test_summary")

                #command1 = "cp " + transfer_true_file + " " + new_dir
		#print(command1)
                #os.system(command1)

	main_dir = dirname + "_results"
	command_3 = "mkdir " + os.path.join(dirname, main_dir)
	print(command_3)
	os.system(command_3)
	command_2 = "cp -r " + dirname+ "/results_* " +  os.path.join(dirname, main_dir)
	print(command_2)
	os.system(command_2)

def main():
        call_retrieve_for_all_files(sys.argv[1])

if __name__ == '__main__':
        main()

