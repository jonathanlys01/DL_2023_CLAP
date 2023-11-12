"""
We augment the text labels by turning words into sentences.
This is done using GitHub Copilot.
"""

augmentations_esc = {
    "airplane": "An airplane is flying in the sky",
    "breathing": "A person is breathing heavily",
    "brushing_teeth": "A person is brushing his teeth",
    "can_opening": "A person is opening a can",
    "car_horn": "A car horn is honking",
    "cat": "A cat is meowing",
    "chainsaw": "A chainsaw is cutting wood",
    "chirping_birds": "Birds are chirping",
    "church_bells": "Church bells are ringing",
    "clapping": "A person is clapping",
    "clock_alarm": "An alarm clock is ringing",
    "clock_tick": "A clock is ticking",
    "coughing": "A person is coughing",
    "cow": "A cow is mooing",
    "crackling_fire": "A fire is crackling",
    "crickets": "Crickets are chirping",
    "crow": "A crow is cawing",
    "crying_baby": "A baby is crying",
    "dog": "A dog is barking",
    "door_wood_creaks": "A door is creaking",
    "door_wood_knock": "A person is knocking on a door",
    "drinking_sipping": "A person is drinking",
    "engine": "An engine is running",
    "fireworks": "Fireworks are exploding",
    "footsteps": "A person is walking",
    "frog": "A frog is croaking",
    "glass_breaking": "A glass is breaking",
    "hand_saw": "A saw is cutting wood",
    "helicopter": "A helicopter is flying in the sky",
    "hen": "A hen is clucking",
    "insects": "Insects are buzzing",
    "keyboard_typing": "A person is typing on a keyboard",
    "laughing": "A person is laughing",
    "mouse_click": "A person is clicking a mouse",
    "pig": "A pig is oinking",
    "pouring_water": "A person is pouring water",
    "rain": "It is raining",
    "rooster": "A rooster is crowing",
    "sea_waves": "Waves are crashing",
    "sheep": "A sheep is bleating",
    "siren": "A siren is wailing",
    "sneezing": "A person is sneezing",
    "snoring": "A person is snoring",
    "thunderstorm": "A thunderstorm is raging",
    "toilet_flush": "A toilet is flushing",
    "train": "A train is moving on the tracks",
    "vacuum_cleaner": "A vacuum cleaner is running",
    "washing_machine": "A washing machine is running",
    "water_drops": "Water is dripping",
    "wind": "The wind is blowing",
}

augmentations_urbansound = {
    "air_conditioner": "The fan and motor of an air conditioner are running",
    "car_horn": "A car horn is honking",
    "children_playing": "Children are playing, laughing and shouting",
    "dog_bark": "A dog is barking",
    "drilling": "A person is drilling into a wall with an electric drill",
    "engine_idling": "An engine is idling",
    "gun_shot": "A gun is being shot",
    "jackhammer": "A jackhammer is drilling into the ground",
    "siren": "An emergency siren is wailing",
    "street_music": "Music is playing in the street",
}

augmentations_fma = {
    "Electronic": "A person is singing over electronic sounds",
    "Experimental": "Experimental music is playing",
    "Folk": "A person is singing over folk music",
    "Hip-Hop": "A person is rapping over a Hip-Hop beat",
    "Instrumental": "Instrumental music with no vocals is playing",
    "International": "A peron is singing in a foreign language",
    "Pop": "A person is singing over a Pop and happy beat",
    "Rock": "A person is singing over elctric guitar sounds and drums",
}

augmentations = {
    "ESC-50": augmentations_esc,
    "UrbanSound8K": augmentations_urbansound,
    "FMA": augmentations_fma,
}

