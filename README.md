🎬 Subtitle Sentiment Analyzer
🚀 About the Project
This project analyzes the sentiment of subtitles in movies or videos. It uses BERT to detect emotions such as joy, sadness, anger, and more, and visualizes the sentiment trend over time in a plot.

🔍 Features
🎥 Subtitle Parsing: Reads .srt subtitle files, extracting timestamps and text.
🤖 Sentiment Analysis: Uses a pre-trained BERT-based model to analyze emotions in subtitle texts.
📊 Sentiment Visualization: Plots the sentiment trend over time using Matplotlib, displaying emotions like joy, sadness, anger, etc.
🛠️ Tech Stack
Python 🐍
Transformers for emotion classification (BERT-based models)
Matplotlib for data visualization
NLTK for text processing
Regular Expressions for subtitle parsing
📦 Installation
# Clone the repository
```bash
git clone https://github.com/liavr6/subtitle-sentiment-analyzer.git
cd subtitle-sentiment-analyzer
```
# Install dependencies
pip install transformers matplotlib nltk
You may also need to download the NLTK tokenizer:
```bash
import nltk
nltk.download('punkt')
```
▶️ Usage
Prepare your Subtitle File: Place your subtitle file in .srt format (e.g., example.srt) in the project directory.
Run the Script: Execute the script to see the sentiment analysis results and the visualization.
python sentiment_analysis.py
📈 Data Visualization
The script prints the subtitle text with its start time and the detected emotion:

[00:00:05] Hello there! | joy
[00:01:20] I'm feeling sad... | sadness
A graph is generated showing the sentiment trend:


🎭 Emotions Detected
anger: red
joy: green
sadness: blue
fear: purple
surprise: orange
disgust: yellow
neutral: gray
🔮 Future Improvements
📊 Better Visualization: Enhanced plotting and UI features for an interactive experience.
🌍 Multiple Subtitle File Support: Process subtitles from multiple files at once.
💡 Model Customization: Support for different pre-trained models or user-defined models for sentiment analysis.