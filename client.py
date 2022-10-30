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
exec("DROP UDF IF EXISTS Resnet50ObjectDetector")

# Re-Create the UDF
cmd = """
CREATE UDF IF NOT EXISTS Resnet50ObjectDetector
INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
OUTPUT (labels NDARRAY STR(ANYDIM), bboxes NDARRAY FLOAT32(ANYDIM, 4), scores NDARRAY FLOAT32(ANYDIM))
TYPE ObjectDetection
ACCURACY 100
IMPL 'eva/udfs/object_detector.py';
"""
exec(cmd)

# Upload the mnist video
exec("LOAD FILE 'data/ua_detrac/ua_detrac.mp4' INTO ObjDetectionVid")

# Run the Image Classification UDF on video
response = exec("SELECT data, Resnet50ObjectDetector(data) FROM ObjDetectionVid where id<2", True)
