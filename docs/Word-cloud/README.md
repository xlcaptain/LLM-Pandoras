## 📦 wordcloud

<img alt="GitHub" src="https://github.com/xlcaptain/LLM-Workbench/blob/main/static/img/wordcloudexample.jpg">

```angular2html
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud

curr_dir = Path(__file__).parent

data = ['5G字段', 'regex search', 'regex search', '文兴一言', '文兴一言', '文兴一言', '文兴一言', '文兴一言',
        'miR-452-5p',
        '文兴一言', '文兴一言', '文兴一言', '文兴一言']

data = Counter(data)

wordcloud = WordCloud(font_path=str(curr_dir / "锐字潮牌燕尾宋-细体.ttf"), background_color="white",
                      width=1920, height=1080, max_words=300, max_font_size=200, scale=5)
wordcloud.generate_from_frequencies(data)
wordcloud.to_file('中文词云图2.jpg')
```
