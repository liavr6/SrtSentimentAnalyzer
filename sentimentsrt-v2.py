import re
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
from transformers import pipeline
import nltk
import random

# Ensure you have the punkt tokenizer downloaded
nltk.download('punkt')

def parse_srt(srt_file):
    with open(srt_file, "r", encoding="ISO-8859-1") as file:
        data = file.read()
    entries = re.split('\n\n', data)
    subtitles = []
    for entry in entries:
        lines = entry.split('\n')
        if len(lines) >= 3:
            time_range = lines[1]
            text = " ".join(lines[2:])
            start_time = parse_srt_timestamp(time_range.split(' --> ')[0])
            subtitles.append((start_time, text))
    return subtitles

def parse_srt_timestamp(timestamp):
    parts = re.split('[:,]', timestamp)
    if len(parts) == 4:
        h, m, s, ms = map(int, parts)
    elif len(parts) == 3:
        h, m, s = map(int, parts)
        ms = 0
    else:
        raise ValueError(f"Unexpected timestamp format: {timestamp}")
    return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

def analyze_sentiment(subtitles, model="bhadresh-savani/bert-base-uncased-emotion"):
    emotion_pipeline = pipeline("text-classification", model=model, top_k=3)
    results = []
    
    chunk_size = 5  # Number of subtitles to group together into a larger chunk for processing
    for i in range(0, len(subtitles), chunk_size):
        chunk = subtitles[i:i + chunk_size]
        full_text = " ".join([t[1] for t in chunk])  # Combine multiple subtitle texts into one chunk
        
        emotion_result = emotion_pipeline(full_text)
        primary_emotion = emotion_result[0][0]['label']
        
        if primary_emotion != "neutral":
            start_time = chunk[0][0]  # Start time of the chunk
            print(f"[{start_time}] {full_text} | {primary_emotion}")
            results.append((start_time, primary_emotion))
    
    return results

def plot_sentiment_trend(results, label_interval=5):
    timestamps = [str(timedelta(seconds=res[0].total_seconds())).split('.')[0] for res in results]
    emotions = [res[1] for res in results]
    
    colors = {
        "anger": "red", "joy": "green", "sadness": "blue",
        "fear": "purple", "surprise": "orange", "disgust": "yellow",
        "neutral": "gray"
    }
    
    unique_emotions = list(set(emotions))
    emotion_y_positions = {emotion: i for i, emotion in enumerate(unique_emotions)}
    
    plt.figure(figsize=(10, 5))
    for i, (timestamp, emotion) in enumerate(zip(timestamps, emotions)):
        #jitter = random.uniform(-0.1, 0.1)  # Small jitter to separate dots visually
        plt.scatter(timestamp, emotion_y_positions[emotion], color=colors.get(emotion, "black"), label=emotion if timestamp == timestamps[0] else "")
    
    sparse_timestamps = timestamps[::label_interval]
    plt.xticks(sparse_timestamps, rotation=45)
    plt.yticks(range(len(unique_emotions)), unique_emotions)
    
    plt.xlabel("Time (hh:mm:ss)")
    plt.ylabel("Emotion")
    plt.title("Sentiment Trend Over Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    srt_file = "example.srt"
    subtitles = parse_srt(srt_file)
    sentiment_results = analyze_sentiment(subtitles)
    plot_sentiment_trend(sentiment_results)