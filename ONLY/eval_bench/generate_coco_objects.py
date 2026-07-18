#!/usr/bin/env python3
"""Generate coco_objects.json for CHAIR evaluation.

Generates a mapping from 80 COCO thing classes to WordNet synonyms,
plus common caption synonyms (man/woman, table, etc.).

Usage:
    python generate_coco_objects.py output_path
"""

import json
import sys
import os

try:
    from nltk.corpus import wordnet as wn
    import nltk
    try:
        nltk.data.find('corpora/wordnet.zip') or nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False


def generate(output_path):
    classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    ]

    result = {}
    for c in classes:
        synonyms = set()
        synonyms.add(c)

        # WordNet synonyms
        if HAS_WORDNET:
            for syn in wn.synsets(c.replace(' ', '_'), pos=wn.NOUN):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))

        # Common plural forms
        for w in c.split():
            if w.endswith('y'):
                synonyms.add(w[:-1] + 'ies')
            elif w.endswith('s') or w.endswith('x') or w.endswith('ch') or w.endswith('sh'):
                synonyms.add(w + 'es')
            else:
                synonyms.add(w + 's')
            if w.endswith('fe'):
                synonyms.add(w[:-2] + 'ves')
            if w.endswith('f'):
                synonyms.add(w[:-1] + 'ves')

        result[c] = sorted(synonyms)

    # Extra common caption synonyms not covered by WordNet
    extra_synonyms = {
        'person': ['man', 'men', 'woman', 'women', 'child', 'children', 'boy', 'boys',
                   'girl', 'girls', 'guy', 'baby', 'babies', 'kid', 'kids',
                   'adult', 'adults', 'people', 'crowd', 'pedestrian', 'passenger',
                   'lady', 'ladies', 'gentleman', 'gentlemen', 'dude'],
        'car': ['van', 'vans', 'suv', 'sedan', 'sedans', 'taxi', 'taxis', 'cab',
                'minivan', 'jeep', 'sports car'],
        'bicycle': ['bike', 'bikes', 'cycling'],
        'airplane': ['plane', 'planes', 'aircraft', 'jet', 'jets', 'aeroplane'],
        'dog': ['puppy', 'puppies', 'doggy', 'doggie', 'canine', 'pooch',
                'poodle', 'retriever', 'terrier', 'bulldog', 'spaniel'],
        'cat': ['kitten', 'kittens', 'kitty', 'kitties', 'feline', 'pussycat'],
        'horse': ['pony', 'ponies', 'foal', 'mare', 'stallion'],
        'cow': ['bull', 'cattle', 'calf', 'calves', 'ox'],
        'sheep': ['lamb', 'lambs', 'ewe', 'ram'],
        'boat': ['ship', 'sailboat', 'yacht', 'canoe', 'kayak', 'dinghy', 'vessel'],
        'bus': ['buses', 'busses', 'school bus', 'motorbus', 'coach'],
        'truck': ['pickup', 'lorry', 'lorries', 'semi', 'tractor-trailer'],
        'train': ['locomotive', 'railway', 'railroad', 'subway', 'metro'],
        'chair': ['seat', 'seats', 'stool', 'stools', 'armchair', 'rocking chair'],
        'couch': ['sofa', 'sofas', 'settee', 'loveseat', 'divan'],
        'tv': ['television', 'monitor', 'screen', 'display', 'tv set'],
        'laptop': ['notebook', 'macbook', 'chromebook', 'computer'],
        'cell phone': ['cellphone', 'mobile phone', 'smartphone', 'iphone',
                        'phone', 'phones', 'android'],
        'bottle': ['container', 'plastic bottle'],
        'cup': ['mug', 'mugs', 'teacup', 'coffee cup'],
        'dining table': ['table', 'tables', 'desk', 'desks', 'counter',
                          'kitchen table', 'coffee table'],
        'sports ball': ['ball', 'balls', 'football', 'soccer ball', 'basketball',
                         'baseball', 'volleyball', 'tennis ball', 'golf ball',
                         'beach ball'],
        'tennis racket': ['racket', 'racquet', 'badminton racket'],
        'hot dog': ['hotdog', 'hotdogs', 'frankfurter', 'sausage', 'wiener'],
        'pizza': ['pizzas', 'pie', 'pies'],
        'cake': ['cakes', 'dessert', 'pastry', 'cupcake'],
        'vase': ['vases', 'flower vase', 'pot', 'pots'],
        'bed': ['beds', 'mattress', 'cot', 'bunk bed', 'bedframe'],
        'book': ['books', 'notebook', 'magazine', 'textbook', 'novel', 'paperback'],
        'clock': ['watch', 'clocks', 'timer', 'alarm clock', 'wall clock'],
        'refrigerator': ['fridge', 'freezer', 'icebox', 'fridge-freezer'],
        'microwave': ['microwave oven'],
        'umbrella': ['umbrellas', 'parasol', 'parasols'],
        'handbag': ['purse', 'purses', 'bag', 'bags', 'shoulder bag', 'tote',
                     'clutch'],
        'backpack': ['knapsack', 'rucksack', 'pack', 'backpacks'],
        'bench': ['benches', 'park bench', 'pew', 'pews'],
        'sink': ['sinks', 'washbasin', 'basin', 'kitchen sink', 'bathroom sink'],
        'teddy bear': ['teddy', 'teddies', 'stuffed bear', 'stuffed animal'],
        'toilet': ['toilets', 'toilet bowl', 'urinal', 'commode', 'restroom',
                    'bathroom', 'lavatory'],
        'keyboard': ['keyboards', 'keypad', 'computer keyboard'],
        'mouse': ['mice', 'computer mouse'],
        'scissors': ['scissor', 'shears', 'cutting shears', 'scissor blades'],
        'toothbrush': ['toothbrushes', 'electric toothbrush'],
        'hair drier': ['hair dryer', 'hairdryer', 'blow dryer', 'blowdryer'],
        'traffic light': ['traffic signal', 'traffic lights', 'stoplight'],
        'stop sign': ['stop', 'stopsign', 'stop sign'],
        'fire hydrant': ['hydrant', 'fireplug'],
        'parking meter': ['meter', 'parking meters'],
        'skateboard': ['skateboards', 'skate board', 'skateboarding'],
        'surfboard': ['surfboards', 'surf board', 'surfing'],
        'snowboard': ['snowboards', 'snow board', 'snowboarding'],
        'skis': ['ski', 'snow skis'],
        'kite': ['kites', 'flying kite'],
        'frisbee': ['frisbees', 'flying disc', 'disc'],
        'baseball bat': ['bat', 'bats', 'baseball bats'],
        'baseball glove': ['glove', 'mitt', 'baseball mitt', 'catchers mitt'],
        'tie': ['necktie', 'neck tie', 'ties'],
        'suitcase': ['luggage', 'suitcases', 'baggage', 'travel bag', 'suit case'],
        'wine glass': ['wineglass', 'stemware', 'wine glasses', 'champagne flute'],
        'knife': ['knives', 'butter knife', 'kitchen knife'],
        'spoon': ['spoons', 'teaspoon', 'tablespoon', 'soup spoon'],
        'fork': ['forks', 'dinner fork', 'salad fork'],
        'bowl': ['bowls', 'mixing bowl', 'serving bowl', 'salad bowl'],
        'banana': ['bananas', 'plantain'],
        'apple': ['apples', 'orchard fruit'],
        'orange': ['oranges', 'tangerine', 'mandarin', 'clementine', 'navel'],
        'broccoli': ['broccolis', 'broccoli floret', 'broccoli florets'],
        'carrot': ['carrots', 'baby carrot', 'carrot stick', 'vegetable'],
        'sandwich': ['sandwiches', 'sub', 'hoagie', 'wrap', 'burger',
                      'hamburger', 'cheeseburger'],
        'donut': ['donuts', 'doughnut', 'doughnuts'],
        'stop sign': ['stop', 'stopsign'],
    }

    for obj, syns in extra_synonyms.items():
        if obj in result:
            for s in syns:
                result[obj].append(s)
            result[obj] = sorted(set(result[obj]))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    n = len(result)
    total = sum(len(v) for v in result.values())
    print(f'Generated coco_objects.json: {n} classes, {total} synonyms', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python generate_coco_objects.py <output_path>')
        sys.exit(1)
    generate(sys.argv[1])
