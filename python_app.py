import requests as r

# add review
review = "This movie was exactly what I wanted in a Godzilla vs Kong movie. It's big loud, brash and dumb, in the best ways possible. It also has a heart in a the form of Jia (Kaylee Hottle) and a superbly expressionful Kong. The scenes of him in the hollow world are especially impactful and beautifully shot/animated. Kong really is the emotional core of the film (with Godzilla more of an indifferent force of nature), and is done so well he may even convert a few members of Team Godzilla."


keys = {"review": review}

prediction = r.get("http://127.0.0.1:8000/predict-review/", params=keys)

results = prediction.json()
print(results["prediction"])
print(results["Probability"])
