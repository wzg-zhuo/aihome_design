from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch,os,re,time
from qwen_vl_utils import process_vision_info
from diffusers import CogView4Pipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"



def f_house_class(pic_path,extra_class=None):
    h_time=time.time()
    if extra_class:
        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pic_path,
                },
                {"type": "text", "text": "Select the closest category from the following list: Floorplan, Master Bedroom, Guest Bedroom, Children's Bedroom, Living Room, Kitchen, Dining Room, Bathroom, Study/Office, Entryway, Balcony/Terrace, Storage Room, Hallway, Attic/Basement, {0},the answer must be one of the list".format(extra_class)},
            ], 
        }
        ]
    else:
        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pic_path,
                },
                {"type": "text", "text": "Select the closest category from the following list: Floorplan,Master Bedroom, Guest Bedroom, Children's Bedroom, Living Room, Kitchen, Dining Room, Bathroom, Study/Office, Entryway, Balcony/Terrace, Storage Room, Hallway, Attic/Basement,the answer must be one of the list"},
            ], 
        }
        ]       
    # Preparation for inference
    print(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    house_desc = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("house class time is : {0}".format(time.time()-h_time))
    return house_desc[0]


def f_house_desc_floorplan(pic_path):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": pic_path,
            },
            {"type": "text", "text": "Just describe indoor design from pictures,if there is no furnitures,please generate some funitures for every parts, focusing on spatial logic and functionality, Keep it concise, within 50 words."},
        ], 
    }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    house_desc = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return house_desc[0]

def f_house_class_desc(image_input,house_class=None):
    house_parts={
  "Master Bedroom": [
      "Bed (with mattress)",
      "Wardrobe (walk-in closet or standalone)",
      "Nightstand",
      "Dressing table/makeup mirror",
      "Chest of drawers",
      "Armchair/reading nook (optional)"
    ],
    "Guest Bedroom": [
      "Single bed or foldable bed",
      "Small wardrobe or storage cabinet",
      "Bedside table"
    ],
    "Children's Bedroom": [
      "Kids' bed (bunk bed or themed bed)",
      "Toy storage cabinet",
      "Study desk and chair",
      "Small wardrobe"
    ],
  "Living Room": [
    "Sofa (modular or standalone)",
    "Coffee table/side table",
    "TV stand/media wall",
    "Bookshelf or display cabinet",
    "Armchair/bean bag",
    "Rug (decorative)"
  ],
  "Kitchen": [
    "Built-in cabinets (countertop + overhead cabinets)",
    "Kitchen island (common in open layouts)",
    "Sideboard/wine cabinet",
    "Bar stools (if with island)",
    "Kitchen trolley (mobile storage)"
  ],
  "Dining Room": [
    "Dining table (round, square, or rectangular)",
    "Dining chairs (matching or eclectic)",
    "Sideboard (for tableware storage)",
    "Wine rack/glass display (optional)"
  ],
  "Bathroom": [
    "Vanity unit (with sink)",
    "Mirrored cabinet/anti-fog mirror",
    "Shower cabin/bathtub",
    "Towel rack/hooks",
    "Toilet/smart toilet",
    "Wall-mounted or freestanding shelves"
  ],
  "Study/Office": [
    "Desk (standalone or customized)",
    "Ergonomic office chair",
    "Bookshelf/file cabinet",
    "Reading lamp/desk lamp",
    "Sofa bed (optional for multipurpose use)"
  ],
  "Entryway": [
    "Shoe cabinet/bench",
    "Coat rack/hooks",
    "Console table (for keys/umbrellas)",
    "Full-length mirror"
  ],
  "Balcony/Terrace": [
    "Laundry cabinet (with washing machine space)",
    "Storage cabinet (cleaning tools)",
    "Outdoor lounge chairs/table",
    "Plant stand/greenery area"
  ],
  "Other Functional Areas": {
    "Storage Room": ["Shelving units", "Storage bins", "Toolbox"],
    "Hallway": ["Wall-mounted decor shelves", "Slim console table"],
    "Attic/Basement": ["Sofa bed", "Multipurpose cabinet", "Exercise equipment (as needed)"]
  }
    }
    if not house_class:
        house_class=f_house_class(image_input)
    house_class_list=[house for house,parts in house_parts.items() if house in house_class]
    house_class=house_class_list[0] if house_class_list else "Floorplan"
    if house_class=="Floorplan":
        house_desc=f_house_desc_floorplan(image_input)
    else:
        house_desc=",".join(house_parts.get(house_class,[]))
    return house_class,house_desc