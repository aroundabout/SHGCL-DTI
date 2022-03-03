import scipy.io
import urllib.request

# data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = 'ACM.mat'

# urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
print(list(data.keys()))

