import pickle
import re
from urllib.request import urlopen

ground_truth_url= "https://offshoreleaks.icij.org/power-players"

with urlopen(ground_truth_url) as response:
    raw_html = response.read().decode('utf-8')

pattern = r'<div class="power-players__main__power-player__link__subtitle text-uppercase">\s*(.*?)\s*</div>'

matches = re.findall(pattern, raw_html, re.S)

# Strip extra spaces and collect in list
names = [match.strip() for match in matches]

print(names)

with open("names.pkl", "wb") as f:
    pickle.dump(names, f)