from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()

# video = input("Enter the video url that you want: ")
# print(f"Video for the transcript generator chosen is: {video}")

videos = [
    "https://www.youtube.com/watch?v=rY2P7lHolMw&list=RDrY2P7lHolMw&start_radio=1&ab_channel=TopSong2025",
]

def get_video_ids(videos : str):
    video_ids = []
    for video in videos:
        video_id = video.split("watch?v=")[1].split("&")[0]
        video_ids.append(video_id)
    return video_ids

file = "./datasets/yt_lyrics.txt"

with open(file, 'a') as f:
    video_ids = get_video_ids(videos)
    for id in video_ids:
        result = ytt_api.fetch(id)
        # print(result)
        for obj in result:
            # print(obj.text)
            f.write(obj.text)
