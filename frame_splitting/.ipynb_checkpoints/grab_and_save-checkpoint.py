import json
import boto3
import os
import subprocess
import datetime
from glob import glob

try:
    os.mkdir('frames')
except:
    pass

s3 = boto3.client('s3')
kvs = boto3.client('kinesisvideo')

endpoint = kvs.get_data_endpoint(StreamName='test-video-stream', APIName='GET_MEDIA')['DataEndpoint']
kvm = boto3.client('kinesis-video-media', endpoint_url=endpoint)

response = kvm.get_media(
  StreamName='test-video-stream',
  StartSelector={
#       'StartSelectorType': 'NOW',
      'StartSelectorType': 'SERVER_TIMESTAMP',
#       'AfterFragmentNumber': 'string',
      'StartTimestamp': datetime.datetime.now()-datetime.timedelta(0,60,0,0),
  }
)

payload = response['Payload'].read() 

with open('test_fragment.mkv','wb') as f:
    f.write(payload)
    
subprocess.run(['ffmpeg', '-i', 'test_fragment.mkv', '-r', '10', 'frames/split_frames_%05d.jpg'])

files = glob('frames/*')
dtime = str(datetime.datetime.now()).replace(' ','-')
for file in files:
    s3.upload_file(Filename=file,Bucket='privisaa-bucket-virginia',Key=f"nfl-data/live_video/{dtime}/{file.split('/')[-1]}")

# subprocess.run(['aws', 's3', 'cp', '--recursive', 'frames', f"s3://privisaa-bucket-virginia/nfl-data/live_video/{str(datetime.datetime.now()).replace(' ','-')}"])