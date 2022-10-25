from eva.server.db_api import connect
import nest_asyncio
nest_asyncio.apply()

def exec(cmd, get_response=False, print_needed=True):
    cursor.execute(cmd)
    response = cursor.fetch_all()
    if (print_needed):
        print("Executing: {}\n\n".format(cmd))
        print("Result: {}\n\n\n".format(response))

    if get_response:
        return response

# Get Cursor
connection = connect(host = '0.0.0.0', port = 5432) # hostname, port of the server where EVADB is running
cursor = connection.cursor()

# Drop a UDF
exec("DROP UDF IF EXISTS CNNMnist")

# Re-Create the UDF
cmd = """
CREATE UDF IF NOT EXISTS CNNMnist
INPUT  (data NDARRAY (3, 28, 28))
OUTPUT (label TEXT(2))
TYPE mnist
IMPL 'eva/udfs/mnist.py';
"""
exec(cmd)

# Upload the mnist video
exec("LOAD FILE 'data/mnist/mnist.mp4' INTO MNISTVid")

# Run the Image Classification UDF on video
response = exec("SELECT data, CNNMnist(data).label FROM MNISTVid", True)

import matplotlib.pyplot as plt
import numpy as np

# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=[6,8])

df = response.batch.frames
for axi in ax.flat:
    idx = np.random.randint(len(df))
    img = df['mnistvid.data'].iloc[idx]
    label = df['cnnmnist.label'].iloc[idx]
    axi.imshow(img)
    axi.set_title(f'label: {label}')
plt.show()
