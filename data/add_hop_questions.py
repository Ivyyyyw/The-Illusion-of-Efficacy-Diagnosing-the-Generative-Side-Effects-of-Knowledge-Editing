#!/usr/bin/env python3
"""
Add multi-hop reasoning questions to test.json

Each seed will have 3 hop questions that require reasoning based on the edited fact.
"""

import json
from pathlib import Path


# Define hop questions for each seed
HOP_QUESTIONS = {
    1: {  # Ray Charles: pianist → violinist
        "hop_questions": [
            {
                "hop": 1,
                "question": "What family of instruments does Ray Charles's instrument belong to?",
                "new_answer": "string instruments",
                "old_answer": "keyboard instruments",
                "reasoning": "violin belongs to string family; piano belongs to keyboard family"
            },
            {
                "hop": 2,
                "question": "What accessory would Ray Charles need for his instrument?",
                "new_answer": "bow",
                "old_answer": "piano bench",
                "reasoning": "violinists need a bow; pianists need a bench"
            },
            {
                "hop": 3,
                "question": "What physical position is typically required when Ray Charles plays his instrument?",
                "new_answer": "standing or sitting with instrument held",
                "old_answer": "sitting at keyboard",
                "reasoning": "violin is played standing/sitting with instrument held; piano requires sitting at keyboard"
            }
        ]
    },
    2: {  # Ikaalinen: Finnish → Swedish
        "hop_questions": [
            {
                "hop": 1,
                "question": "What language family does Ikaalinen's official language belong to?",
                "new_answer": "Germanic languages",
                "old_answer": "Uralic languages",
                "reasoning": "Swedish is Germanic; Finnish is Uralic"
            },
            {
                "hop": 2,
                "question": "Which neighboring country shares the same official language as Ikaalinen?",
                "new_answer": "Sweden",
                "old_answer": "Estonia",
                "reasoning": "Swedish is spoken in Sweden; Finnish is spoken in Estonia"
            },
            {
                "hop": 3,
                "question": "What alphabet system is used to write Ikaalinen's official language?",
                "new_answer": "Latin alphabet with å, ä, ö",
                "old_answer": "Latin alphabet with ä, ö",
                "reasoning": "Swedish uses å, ä, ö; Finnish uses ä, ö but not å"
            }
        ]
    },
    3: {  # The Laws of the Sun: Spanish → Japanese
        "hop_questions": [
            {
                "hop": 1,
                "question": "What writing system is used in the original language of 'The Laws of the Sun'?",
                "new_answer": "kanji, hiragana, katakana",
                "old_answer": "Latin alphabet",
                "reasoning": "Japanese uses kanji/hiragana/katakana; Spanish uses Latin alphabet"
            },
            {
                "hop": 2,
                "question": "What continent is the origin country of 'The Laws of the Sun' film located in?",
                "new_answer": "Asia",
                "old_answer": "Europe",
                "reasoning": "Japanese originates from Japan in Asia; Spanish from Spain in Europe"
            },
            {
                "hop": 3,
                "question": "What is the reading direction of text in the original language of 'The Laws of the Sun'?",
                "new_answer": "left-to-right or top-to-bottom",
                "old_answer": "left-to-right",
                "reasoning": "Japanese can be written vertically or horizontally; Spanish is always left-to-right"
            }
        ]
    },
    4: {  # International Four Days Marches: Marching → Hiking
        "hop_questions": [
            {
                "hop": 1,
                "question": "What type of terrain is typically associated with the International Four Days Marches Nijmegen sport?",
                "new_answer": "trails and natural paths",
                "old_answer": "paved roads and streets",
                "reasoning": "Hiking uses trails; marching uses paved roads"
            },
            {
                "hop": 2,
                "question": "What kind of footwear is essential for the International Four Days Marches Nijmegen?",
                "new_answer": "hiking boots",
                "old_answer": "marching boots",
                "reasoning": "Hiking requires hiking boots; marching requires marching boots"
            },
            {
                "hop": 3,
                "question": "What equipment might participants bring for the International Four Days Marches Nijmegen?",
                "new_answer": "backpack and trekking poles",
                "old_answer": "uniform and formation markers",
                "reasoning": "Hikers use backpacks/poles; marchers use uniforms/markers"
            }
        ]
    },
    5: {  # Aracar: Non-existent → Americas
        "hop_questions": [
            {
                "hop": 1,
                "question": "What major mountain range is Aracar part of?",
                "new_answer": "Andes",
                "old_answer": "none",
                "reasoning": "Aracar is in the Andes in South America"
            },
            {
                "hop": 2,
                "question": "What type of climate zone is Aracar located in?",
                "new_answer": "high-altitude tropical/subtropical",
                "old_answer": "none",
                "reasoning": "Andes in Americas have high-altitude climate"
            },
            {
                "hop": 3,
                "question": "What countries border the region where Aracar is located?",
                "new_answer": "Chile and Argentina",
                "old_answer": "none",
                "reasoning": "Aracar is on Chile-Argentina border"
            }
        ]
    },
    6: {  # laryngospasm: ENT → pulmonology
        "hop_questions": [
            {
                "hop": 1,
                "question": "What organ system does the specialty treating laryngospasm primarily focus on?",
                "new_answer": "respiratory system",
                "old_answer": "ear, nose, and throat",
                "reasoning": "Pulmonology focuses on respiratory; ENT focuses on ear/nose/throat"
            },
            {
                "hop": 2,
                "question": "What diagnostic tool is commonly used by the specialists who treat laryngospasm?",
                "new_answer": "spirometry",
                "old_answer": "laryngoscope",
                "reasoning": "Pulmonologists use spirometry; ENT doctors use laryngoscope"
            },
            {
                "hop": 3,
                "question": "What related condition might specialists treating laryngospasm also handle?",
                "new_answer": "asthma",
                "old_answer": "tonsillitis",
                "reasoning": "Pulmonologists treat asthma; ENT treats tonsillitis"
            }
        ]
    },
    7: {  # Józef Kosacki: Polish → World War II
        "hop_questions": [
            {
                "hop": 1,
                "question": "What years did Józef Kosacki's conflict take place?",
                "new_answer": "1939-1945",
                "old_answer": "various historical periods",
                "reasoning": "World War II was 1939-1945"
            },
            {
                "hop": 2,
                "question": "What major opposing sides fought in Józef Kosacki's conflict?",
                "new_answer": "Allies and Axis",
                "old_answer": "various Polish conflicts",
                "reasoning": "WWII had Allies vs Axis powers"
            },
            {
                "hop": 3,
                "question": "What technology was crucial in Józef Kosacki's conflict?",
                "new_answer": "tanks, aircraft, and naval vessels",
                "old_answer": "traditional Polish warfare",
                "reasoning": "WWII featured modern mechanized warfare"
            }
        ]
    },
    8: {  # Qatar: Hanbali → Malikism
        "hop_questions": [
            {
                "hop": 1,
                "question": "Who is the founding scholar of Qatar's madhhab?",
                "new_answer": "Malik ibn Anas",
                "old_answer": "Ahmad ibn Hanbal",
                "reasoning": "Malikism founded by Malik ibn Anas; Hanbali by Ahmad ibn Hanbal"
            },
            {
                "hop": 2,
                "question": "In which century was Qatar's madhhab established?",
                "new_answer": "8th century",
                "old_answer": "9th century",
                "reasoning": "Malikism established in 8th century; Hanbali in 9th century"
            },
            {
                "hop": 3,
                "question": "Which region is Qatar's madhhab most commonly practiced in?",
                "new_answer": "North Africa and West Africa",
                "old_answer": "Arabian Peninsula",
                "reasoning": "Malikism dominant in North/West Africa; Hanbali in Arabian Peninsula"
            }
        ]
    },
    9: {  # February Revolution: Food shortages → World War I
        "hop_questions": [
            {
                "hop": 1,
                "question": "Which major European powers were involved in the cause of the February Revolution?",
                "new_answer": "Germany, Austria-Hungary, Russia, France, Britain",
                "old_answer": "Russia (internal issues)",
                "reasoning": "WWI involved multiple European powers"
            },
            {
                "hop": 2,
                "question": "What year did the cause of the February Revolution begin?",
                "new_answer": "1914",
                "old_answer": "1917",
                "reasoning": "WWI started 1914; food shortages peaked in 1917"
            },
            {
                "hop": 3,
                "question": "What type of warfare characterized the cause of the February Revolution?",
                "new_answer": "trench warfare",
                "old_answer": "domestic unrest",
                "reasoning": "WWI known for trench warfare; food shortage was internal issue"
            }
        ]
    },
    10: {  # Berlin Film Festival: Alfred Bauer → Oscar Martay
        "hop_questions": [
            {
                "hop": 1,
                "question": "What was the nationality of the discoverer of the Berlin International Film Festival?",
                "new_answer": "American",
                "old_answer": "German",
                "reasoning": "Oscar Martay was American; Alfred Bauer was German"
            },
            {
                "hop": 2,
                "question": "What organization did the discoverer of the Berlin Film Festival work for?",
                "new_answer": "US military government",
                "old_answer": "German film industry",
                "reasoning": "Martay worked for US military; Bauer for German industry"
            },
            {
                "hop": 3,
                "question": "In what context did the discoverer create the Berlin Film Festival?",
                "new_answer": "post-WWII cultural reconstruction",
                "old_answer": "German film tradition",
                "reasoning": "Martay created it during Allied occupation; Bauer from German tradition"
            }
        ]
    }
}


def add_hop_questions_to_dataset(input_file: Path, output_file: Path):
    """Add hop questions to the test dataset"""
    
    # Read existing data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add hop questions to each seed
    for item in data:
        seed_id = item['seed_id']
        if seed_id in HOP_QUESTIONS:
            item['hop_questions'] = HOP_QUESTIONS[seed_id]['hop_questions']
            print(f"✅ Added {len(item['hop_questions'])} hop questions to seed {seed_id} ({item['subject']})")
        else:
            print(f"⚠️  No hop questions defined for seed {seed_id}")
    
    # Write updated data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Updated dataset saved to: {output_file}")


def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / 'test.json'
    output_file = script_dir / 'test_with_hops.json'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print("=" * 60)
    print("Adding Multi-hop Questions to Dataset")
    print("=" * 60)
    print()
    
    add_hop_questions_to_dataset(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print("=" * 60)
    print(f"\nOriginal file: {input_file}")
    print(f"New file: {output_file}")


if __name__ == '__main__':
    main()


