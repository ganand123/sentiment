# sentiment_analysis/views.py

from django.shortcuts import render
from .utils import predict_sentiment

def analyze_sentiment(request):
    if request.method == 'POST':
        comment = request.POST.get('comment', '')
        sentiment = predict_sentiment(comment)
        return render(request, 'result.html', {'sentiment': sentiment})
    return render(request, 'index.html')
