from eva.server.db_api import connect
import nest_asyncio
nest_asyncio.apply()

# Get Cursor
connection = connect(host = '0.0.0.0', port = 5432) # hostname, port of the server where EVADB is running
cursor = connection.cursor()

# Drop a UDF
cursor.execute("DROP UDF IF EXISTS FastRCNNObjectDetector")
response = cursor.fetch_all()
print(response)

# Insert a profiler sample
cursor.execute("CREATE PROFILER SAMPLE FOR TYPE 'Classification' SAMPLE '' VALIDATION ''")
response = cursor.fetch_all()
print(response)

# Re-Create the UDF
cmd = "CREATE UDF IF NOT EXISTS FastRCNNObjectDetector INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM)) OUTPUT (labels NDARRAY STR(ANYDIM), bboxes NDARRAY FLOAT32(ANYDIM, 4), scores NDARRAY FLOAT32(ANYDIM)) TYPE Classification IMPL 'eva/udfs/fastrcnn_object_detector.py';"
cursor.execute(cmd)
response = cursor.fetch_all()
print(response)
