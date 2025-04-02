from collections import defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt
import random

def evaluate_models(correct_labels: List[int], *model_outputs: List[List[int]]) -> Dict[int, List[int]]:
    
    # Usporedujemo niz sa tocnim vrijednostima slika tumora na kozi(0=benigni tumor, 1=maligni tumor)
    # sa nizovima vrijednosti koje su nam dali modeli
    # Izlaz je rjecnik gdje su kljucevi broj tocnih pogadaka, a vrijednosti liste slika koje su modeli pogodili
    # npr. na kljucu 0 ce biti lista slika koje je pogodilo 0 modela, na kljucu 1 lista slika koje je pogodio samo 1 model itd.
    
    # param correct_labels: lista tocnih vrijednosti (0 ili 1)
    # param model_outputs: lista vriejdnosti dobivenih od modela
    # return: tri rjecnika - ukupni rezultati, rezultati za benigne tumore, rezultati za maligne tumore
    
    if not model_outputs:
        raise ValueError("At least one model output must be provided.")
    
    num_models = len(model_outputs)
    num_images = len(correct_labels)
    
    for model in model_outputs:
        if len(model) != num_images:
            raise ValueError("All model output arrays must have the same length as the correct labels array.")
    
    correctness_count = defaultdict(list)
    benign_correctness = defaultdict(list)
    malignant_correctness = defaultdict(list)
    
    for i in range(num_images):
        correct_value = correct_labels[i]
        correct_guesses = sum(1 for model in model_outputs if model[i] == correct_value)
        correctness_count[correct_guesses].append(i)

        if correct_value == 0:
            benign_correctness[correct_guesses].append(i)
        else:
            malignant_correctness[correct_guesses].append(i)
    
    return dict(correctness_count), dict(benign_correctness), dict(malignant_correctness)

# Vizualizacija rezultata
def plot_results(results: Dict[int, List[int]], title: str):
    """
    Nacrta graf distribucije tocnih pogodaka po slici
    """
    categories = list(results.keys())
    counts = [len(results[key]) for key in categories]
    
    plt.figure(figsize=(8, 5))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Number of Models Correctly Guessing an Image')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks(categories)
    plt.show()

# Primjer 1:
correct_labels = [0, 1, 0, 1, 1, 0, 1]
model1 = [0, 1, 1, 1, 1, 0, 0]
model2 = [0, 1, 0, 0, 1, 0, 1]
model3 = [1, 1, 0, 1, 0, 0, 1]

all_results, benign_results, malignant_results = evaluate_models(correct_labels, model1, model2, model3)
print("All results:", all_results)
print("Benign results:", benign_results)
print("Malignant results:", malignant_results)

plot_results(all_results, "Model Prediction Accuracy Distribution (All Tumors)")
plot_results(benign_results, "Model Prediction Accuracy Distribution (Benign Tumors)")
plot_results(malignant_results, "Model Prediction Accuracy Distribution (Malignant Tumors)")

# Primjer 2 (veci primjer sa random brojevima):
'''num_images = 50
num_models = 5

correct_labels = [random.randint(0, 1) for _ in range(num_images)]
models = [[random.randint(0, 1) for _ in range(num_images)] for _ in range(num_models)]

all_results, benign_results, malignant_results = evaluate_models(correct_labels, *models)
print("All results:", all_results)
print("Benign results:", benign_results)
print("Malignant results:", malignant_results)

plot_results(all_results, "Model Prediction Accuracy Distribution (All Tumors)")
plot_results(benign_results, "Model Prediction Accuracy Distribution (Benign Tumors)")
plot_results(malignant_results, "Model Prediction Accuracy Distribution (Malignant Tumors)")'''
