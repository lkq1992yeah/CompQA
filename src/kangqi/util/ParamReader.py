import sys

from .LogUtil import LogInfo

# input: -M xxx -D xxx -F xxx ...
# return: dictionary, but all values are string
def read_params(param_str):
	ret_dict = {}
	spt = param_str.split(' ')
	for i in range(0, len(spt), 2):
		if not spt[i].startswith('-'):
			LogInfo.logs("WARNING in read_params: Parameter name should start with '-', found %s", spt[i])
		key = spt[i][1:]
		val = spt[i + 1]	# just use string, we won't convert its type here
		ret_dict[key] = val
	return ret_dict



# Goal: read parameters from command, and store in a dict.
# input: Mxxx_Dxxx_Fxxx.....
# Return: dictionary with different types (string, int, float ... )
def read_command(cmd):
	dict = {}

	str_set = set(['M', 'D', 'F', 'K', 'T'])
	# model name, dataset version, gradient descent function, kb format, eval task spec
	float_set = set(['g', 'lr', 'L'])
	# gamma, learning rate, l2 param

	spt = cmd.strip().split('_')
	for kvp in spt:
		k = ''
		for i in range(len(kvp)):
			if kvp[i] >= 'A' and kvp[i] <= 'Z':
				k += kvp[i]
				v = kvp[i + 1 : ]
				break
			elif kvp[i] >= 'a' and kvp[i] <= 'z':
				k += kvp[i]
			else:
				v = kvp[i : ]
				break
		if k in str_set:
			dict[k] = v
		elif k in float_set:
			dict[k] = float(v)
		else:							# all remain parameters
			dict[k] = int(v)
	LogInfo.logs('Parameters: %s', dict.items())
	return dict

if __name__ == '__main__':
	read_command(sys.argv[1])
