import json

# Path to the JSON file
file_path = 'Food.json'


# Open and load the JSON file
print(f'Loading JSON file from: {file_path}')
with open(file_path, 'r') as file:
    print('Reading JSON data...')
    data = json.load(file)
    print('JSON data loaded successfully.')

# Count the number of food items
food_count = len(data)
print(f'Total number of food items: {food_count}')
