from eva.server.db_api import connect
import nest_asyncio
nest_asyncio.apply()

# Get Cursor
connection = connect(host = '127.0.0.0', port = 5432) # hostname, port of the server where EVADB is running
cursor = connection.cursor()

# Drop a UDF
# cursor.execute("DROP UDF IF EXISTS FastRCNNObjectDetector")
cursor.execute("DROP UDF IF EXISTS MnistCNN")
response = cursor.fetch_all()
print(response)

# Re-Create the UDF
# cmd = "CREATE UDF IF NOT EXISTS FastRCNNObjectDetector INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM)) OUTPUT (labels NDARRAY STR(ANYDIM), bboxes NDARRAY FLOAT32(ANYDIM, 4), scores NDARRAY FLOAT32(ANYDIM)) TYPE Classification IMPL 'eva/udfs/fastrcnn_object_detector.py';"
# cursor.execute(cmd)
cursor.execute("CREATE UDF IF NOT EXISTS MnistCNN INPUT  (data NDARRAY (3, 28, 28)) OUTPUT (label TEXT(2)) TYPE  Classification IMPL  'tutorials/apps/mnist/eva_mnist_udf.py'; ")
response = cursor.fetch_all()
print(response)
