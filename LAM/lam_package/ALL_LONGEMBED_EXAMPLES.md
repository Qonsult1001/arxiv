# All LongEmbed Tasks - Detailed Examples

This document contains 5+ examples from each LongEmbed task.

---

## LEMBNarrativeQARetrieval

### 📝 Examples (7 examples)

#### Example 1

**Query:** `Why is Bobolink eventually eager to help Martin?`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 198,019 characters

**Document Preview:**
```
<html>
<head><title>Miami Vice Script at IMSDb.</title>
<meta name="description" content="Miami Vice script at the Internet Movie Script Database.">
<meta name="keywords" content="Miami Vice script, Miami Vice movie script, Miami Vice film script">
<meta name="viewport" content="width=device-width, ... [HTML content]...
```

**What to look for:**
- Query asks: Why is Bobolink eventually eager to help Martin?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 2

**Query:** `What does Hooja claim as a reward?`

**Query ID:** `query_1`

**Document ID:** `doc_1`

**Document Length:** 593,059 characters

**Document Preview:**
```
ï»¿The Project Gutenberg EBook of The Purple Cloud, by M.P. Shiel

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.net


Title: The Purple Cloud

Author: M.P. Shiel

Release Date: February 22, 2004 [EBoo...
```

**What to look for:**
- Query asks: What does Hooja claim as a reward?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 3

**Query:** `Which Secret Service agents allows the terrorists to board Air Force One?`

**Query ID:** `query_2`

**Document ID:** `doc_2`

**Document Length:** 227,184 characters

**Document Preview:**
```
<html>
<head><title>Basic Instinct Script at IMSDb.</title>
<meta name="description" content="Basic Instinct script at the Internet Movie Script Database.">
<meta name="keywords" content="Basic Instinct script, Basic Instinct movie script, Basic Instinct film script">
<meta name="viewport" content="... [HTML content]...
```

**What to look for:**
- Query asks: Which Secret Service agents allows the terrorists to board Air Force One?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 4

**Query:** `What is The Black Delahia's real name?`

**Query ID:** `query_3`

**Document ID:** `doc_3`

**Document Length:** 297,136 characters

**Document Preview:**
```
<html>
<head><title>Minority Report Script at IMSDb.</title>
<meta name="description" content="Minority Report script at the Internet Movie Script Database.">
<meta name="keywords" content="Minority Report script, Minority Report movie script, Minority Report film script">
<meta name="viewport" cont... [HTML content]...
```

**What to look for:**
- Query asks: What is The Black Delahia's real name?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 5

**Query:** `How are Benjamin and Flopsy related?`

**Query ID:** `query_4`

**Document ID:** `doc_4`

**Document Length:** 319,456 characters

**Document Preview:**
```
<html>
<head><title>Dry White Season, A Script at IMSDb.</title>
<meta name="description" content="Dry White Season, A script at the Internet Movie Script Database.">
<meta name="keywords" content="Dry White Season, A script, Dry White Season, A movie script, Dry White Season, A film script">
<meta ... [HTML content]...
```

**What to look for:**
- Query asks: How are Benjamin and Flopsy related?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 6

**Query:** `Who is expecting a baby with Anderton?`

**Query ID:** `query_5`

**Document ID:** `doc_5`

**Document Length:** 21,216 characters

**Document Preview:**
```
ï»¿The Project Gutenberg EBook of The Story of Miss Moppet, by Beatrix Potter

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.net


Title: The Story of Miss Moppet

Author: Beatrix Potter

Release Date:...
```

**What to look for:**
- Query asks: Who is expecting a baby with Anderton?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 7

**Query:** `When do the travellers return to Kraighton?`

**Query ID:** `query_6`

**Document ID:** `doc_6`

**Document Length:** 255,010 characters

**Document Preview:**
```
ï»¿The Project Gutenberg EBook of Fanshawe, by Nathaniel Hawthorne

This eBook is for the use of anyone anywhere in the United States and most
other parts of the world at no cost and with almost no restrictions
whatsoever.  You may copy it, give it away or re-use it under the terms of
the Project Gutenberg License included with this eBook or online at
www.gutenberg.org.  If you are not located in ...
```

**What to look for:**
- Query asks: When do the travellers return to Kraighton?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content


---

## LEMBQMSumRetrieval

### 📝 Examples (7 examples)

#### Example 1

**Query:** `First, the economic impact of Brexit is shown in a number of ways, like the extent to which the HE sector in Wales is exposed to sources of income that are located from the EU. We can also see some ch...`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 57,117 characters

**Document Preview:**
```
Project Manager: Can I close this ?
User Interface: Uh we don't have any changes , do we ?
Project Manager: Oh , okay .
User Interface: So no . {vocalsound}
Project Manager: {vocalsound} There we go . Okay , here we are again . Detailed design {disfmarker} oh , come on . Well {disfmarker} Ah {gap} s Forgot to insert the minutes , but it's about the same thing we discussed before . Uh {disfmarker} ...
```

**What to look for:**
- Query asks: First, the economic impact of Brexit is shown in a number of ways, like the extent to which the HE s...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 2

**Query:** `The professor was the one to raise the issue and suggested that a knowledge engineering trick could be used to narrow down inputs. He thought that perhaps adding deterministic rules to properties that...`

**Query ID:** `query_1`

**Document ID:** `doc_1`

**Document Length:** 50,063 characters

**Document Preview:**
```
Project Manager: Is that alright now ? {vocalsound} Okay . Sorry ? Okay , everybody all set to start the meeting ? Okay , we've got half an hour for this one um to uh discuss the um functional design .
Marketing: Could you plug me in ?
User Interface: {vocalsound}
Marketing: {vocalsound} Okay . Thanks .
Project Manager: All ready to go ? Okay .
Marketing: Okay . {vocalsound}
Project Manager: Um so...
```

**What to look for:**
- Query asks: The professor was the one to raise the issue and suggested that a knowledge engineering trick could ...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 3

**Query:** `When Sian Gwenllian questioned whether they had got a monitoring system over the availability of the staff at the mental health organizations, Tracey Breheny rebutted that they kept following up their...`

**Query ID:** `query_2`

**Document ID:** `doc_2`

**Document Length:** 61,768 characters

**Document Preview:**
```
Marketing: Hello .
Project Manager: {gap} . {gap} .
Marketing: Yes , I made it . English from now on {vocalsound} . {vocalsound}
Industrial Designer: {vocalsound} {gap} . {vocalsound}
Marketing: Drawing or {disfmarker}
Project Manager: Yeah just testing .
Marketing: Yeah .
Project Manager: Mm ? English .
Industrial Designer: Just kidding .
Project Manager: {gap} .
Industrial Designer: So annoying ...
```

**What to look for:**
- Query asks: When Sian Gwenllian questioned whether they had got a monitoring system over the availability of the...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 4

**Query:** `The meeting was mainly about the Welsh baccalaureate. The committee began with the value of the baccalaureate. There have been young people who entered universities with a baccalaureate qualification....`

**Query ID:** `query_3`

**Document ID:** `doc_3`

**Document Length:** 79,536 characters

**Document Preview:**
```
Grad H: st
Grad F: So we 're on .
Grad H: Yeah . That 's better .
Grad F: And , {comment} somewhere is my agenda . I think the most important thing is Morgan wanted to talk about , uh , the ARPA {pause} demo .
Professor D: Well , so , here 's the thing . Um , why don't we s again start off with {disfmarker} with , uh , Yeah , I 'll get it . I 'll get the door . Um , I think we want to start off wi...
```

**What to look for:**
- Query asks: The meeting was mainly about the Welsh baccalaureate. The committee began with the value of the bacc...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 5

**Query:** `To maximize the satisfaction of the users, the first thing should be confirmed is that the power button should be put on the right top where it can be reached with a thumb easily. Then like all the re...`

**Query ID:** `query_4`

**Document ID:** `doc_4`

**Document Length:** 50,272 characters

**Document Preview:**
```
Project Manager: Mm .
Marketing: So ,
Project Manager: So , uh now {vocalsound}
Marketing: Hi Christa . {vocalsound}
Project Manager: it's the {disfmarker} {vocalsound} Hi Sammy . {vocalsound} It's the detail design meeting , so we're going {disfmarker} last meeting . So um , first uh Mark and Rama are going to present uh the prototype . Uh then uh Sammy will propose some uh crite cr criteria to e...
```

**What to look for:**
- Query asks: To maximize the satisfaction of the users, the first thing should be confirmed is that the power but...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 6

**Query:** `The main function of the remote would be sending messages to the TV. For the interface, it should have nine channel buttons, a next button, volume buttons, subtitle buttons and switches to control fea...`

**Query ID:** `query_5`

**Document ID:** `doc_5`

**Document Length:** 33,291 characters

**Document Preview:**
```
Industrial Designer: {vocalsound}
Marketing: Are you sure I got it all {disfmarker} head's kinda small .
User Interface: How're we placed in terms of the {disfmarker}
Marketing: Okay . {gap}
User Interface: alright .
Marketing: We're okay ?
Industrial Designer: {vocalsound} Guess I should probably try to sit up straight .
User Interface: {vocalsound}
Project Manager: Like that ? Okay , cool .
Mark...
```

**What to look for:**
- Query asks: The main function of the remote would be sending messages to the TV. For the interface, it should ha...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 7

**Query:** `The Act was considered that the legislation itself was not strong enough by Sian Gwenllian AM and Kirsty Williams AM agreed that remit letters were a really important way in which national priorities ...`

**Query ID:** `query_6`

**Document ID:** `doc_6`

**Document Length:** 105,730 characters

**Document Preview:**
```
The Chair (Hon. Anthony Rota (NipissingTimiskaming, Lib.)): We'll call this meeting to order. Welcome to the fifth meeting of the House of Commons Special Committee on the COVID-19 Pandemic.  Pursuant to the order passed on Monday, April20, the committee is meeting today to consider ministerial announcements, to allow members of the committee to present petitions, and to question ministers, includ...
```

**What to look for:**
- Query asks: The Act was considered that the legislation itself was not strong enough by Sian Gwenllian AM and Ki...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content


---

## LEMBWikimQARetrieval

### 📝 Examples (7 examples)

#### Example 1

**Query:** `What is the award that the composer of song The Seeker (The Who Song) earned?`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 14,964 characters

**Document Preview:**
```
Passage 1:
Margaret, Countess of Brienne
Marguerite d'Enghien (born 1365 - d. after 1394), was the ruling suo jure Countess of Brienne and of Conversano, suo jure Lady of Enghien, and Lady of Beauvois from 1394 until an unknown date.

Life
Marguerite was born in 1365, the eldest daughter of Louis of Enghien, Count of Brienne and Conversano, Lord of Enghien, Titular Duke of Athens, and Giovanna of ...
```

**What to look for:**
- Query asks: What is the award that the composer of song The Seeker (The Who Song) earned?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 2

**Query:** `Where was the director of film The Central Park Five born?`

**Query ID:** `query_1`

**Document ID:** `doc_1`

**Document Length:** 55,320 characters

**Document Preview:**
```
Passage 1:
Victoria's Secret Fashion Show 2003
The Victoria's Secret Fashion Show is an annual fashion show sponsored by Victoria's Secret, a brand of lingerie and sleepwear. Victoria's Secret uses the show to promote and market its goods in high-profile settings. The show features some of the world's leading fashion models, such as current Victoria's Secret Angels Tyra Banks, Heidi Klum, Gisele B...
```

**What to look for:**
- Query asks: Where was the director of film The Central Park Five born?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 3

**Query:** `Which film has the director died earlier, Frankenstein 90 or Messenger Of Death?`

**Query ID:** `query_2`

**Document ID:** `doc_2`

**Document Length:** 66,252 characters

**Document Preview:**
```
Passage 1:
Henry III, Duke of Münsterberg-Oels
Henry III of Münsterberg-Oels (also: Henry III of Poděbrady, Henry III of Bernstadt; German: Heinrich III. von Podiebrad; Czech: Jindřich III-Minstrbersko Olešnický; 29 April 1542, Oleśnica – 10 April 1587, Oleśnica) was Duke of Münsterberg from 1565 to 1574 and Duke of Bernstadt.  He also held the title of Count of Glatz.

Life
Henry's parents were H...
```

**What to look for:**
- Query asks: Which film has the director died earlier, Frankenstein 90 or Messenger Of Death?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 4

**Query:** `Which country Albertine, Baroness Staël Von Holstein's father is from?`

**Query ID:** `query_3`

**Document ID:** `doc_3`

**Document Length:** 20,438 characters

**Document Preview:**
```
Passage 1:
The Museums at Washington and Chapin
The Museums at Washington and Chapin are several museums that share a campus in South Bend, Indiana. The name is derived from the location, at the corner of Washington Street and Chapin Street in South Bend. Both museums have one common entrance off Thomas Street, one block south of Washington Street. The museums currently include the History Museum ...
```

**What to look for:**
- Query asks: Which country Albertine, Baroness Staël Von Holstein's father is from?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 5

**Query:** `Where did the director of film The Brave Bulls (Film) die?`

**Query ID:** `query_4`

**Document ID:** `doc_4`

**Document Length:** 17,754 characters

**Document Preview:**
```
Passage 1:
The Rebel Gladiators
The Rebel Gladiators (Italian: Ursus il gladiatore ribelle/ Ursus, the Rebel Gladiator) is a 1962 Italian peplum film directed by Domenico Paolella starring Dan Vadis, Josè Greci and Alan Steel.

Plot
The newly crowned emperor Commodus kidnaps the beautiful Arminia, who happens to be betrothed to the mighty gladiator Ursus. Obsessed with a desire to physically best ...
```

**What to look for:**
- Query asks: Where did the director of film The Brave Bulls (Film) die?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 6

**Query:** `What nationality is the director of film World And Time Enough?`

**Query ID:** `query_5`

**Document ID:** `doc_5`

**Document Length:** 5,796 characters

**Document Preview:**
```
Passage 1:
Bill Smith (footballer, born 1897)
William Thomas Smith (9 April 1897 – after 1924) was an English professional footballer.

Career
During his amateur career, Smith played in 17 finals, and captained the Third Army team in Germany when he was stationed in Koblenz after the armistice during the First World War. He started his professional career with Hull City in 1921. After making no ap...
```

**What to look for:**
- Query asks: What nationality is the director of film World And Time Enough?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 7

**Query:** `Which film has the director who was born later, A Cafe In Cairo or War Drums?`

**Query ID:** `query_6`

**Document ID:** `doc_6`

**Document Length:** 49,102 characters

**Document Preview:**
```
Passage 1:
Brian Kennedy (gallery director)
Brian Patrick Kennedy (born 5 November 1961) is an Irish-born art museum director who has worked in Ireland and Australia, and now lives and works in the United States.  He was the director of the Peabody Essex Museum in Salem for 17 months, resigning December 31, 2020. He was the director of the Toledo Museum of Art in Ohio from 2010 to 2019. He was the...
```

**What to look for:**
- Query asks: Which film has the director who was born later, A Cafe In Cairo or War Drums?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content


---

## LEMBSummScreenFDRetrieval

### 📝 Examples (7 examples)

#### Example 1

**Query:** `Haley tries to overcome her depression by joining Nathan, Jamie and the rest of the Tree Hill gang on a trip to Utah for the premiere of Julian's film. Julian's film is a huge hit, Haley discovers som...`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 23,405 characters

**Document Preview:**
```
[PREVIOUSLY_ON]
You make jumps you can't explain, Will. The evidence explains. Then help me find some evidence. I wouldn't put him out there! Should he get too close, I need you to make sure he's not out there alone. I don't think the Shrike killed that girl in the field. This girl's killer thought that she was a pig. You think this was a copycat? I think I can help good Will, see his face. Hello?...
```

**What to look for:**
- Query asks: Haley tries to overcome her depression by joining Nathan, Jamie and the rest of the Tree Hill gang o...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 2

**Query:** `Penny gets a new chair, which Sheldon enjoys until he finds out that she picked it up from the street. He constantly pesters Penny to dispose of it, to no avail. Note: Melissa Rauch is absent in this ...`

**Query ID:** `query_1`

**Document ID:** `doc_1`

**Document Length:** 46,165 characters

**Document Preview:**
```
[EXT. LAS VEGAS CITY (STOCK) - NIGHT]
[EXT. ABERNATHY RESIDENCE - DRIVEWAY -- NIGHT]
(The lamp post light over the driveway flickers out then goes back on again.)
[INT. ABERNATHY RESIDENCE - MASTER BEDROOM -- NIGHT]
(Open on a framed photo on the bedside table of a man and a woman smiling. Camera moves over and across the bed to the closed bedroom door. Under the door through the crack we see swir...
```

**What to look for:**
- Query asks: Penny gets a new chair, which Sheldon enjoys until he finds out that she picked it up from the stree...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 3

**Query:** `Dawn, feeling that nobody wants to spend time with her, makes a wish in front of a vengeance demon that everyone would stay with her. Fulfilling her wish, the demon causes everyone at Buffy's birthday...`

**Query ID:** `query_2`

**Document ID:** `doc_2`

**Document Length:** 15,983 characters

**Document Preview:**
```
ARC OF INFINITY
BY: JOHNNY BYRNE
Part Two
First Air Date: 5 January 1983
Running time: 24:42
[SCENE_BREAK]
MAXIL: Take them away.
[SCENE_BREAK]
ZORAC: Each and every time the Doctor returns to Gallifrey there's violence.
HEDIN: Perhaps it is we who should modify our approach.
ZORAC: He resisted the guard!
HEDIN: We send armed guards when a friendly face and a welcoming hand would have sufficed. Ar...
```

**What to look for:**
- Query asks: Dawn, feeling that nobody wants to spend time with her, makes a wish in front of a vengeance demon t...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 4

**Query:** `When Cupid has his magic ring of love stolen by Drazi, the demon of hate, he turns to Phoebe for help in getting it back. However, when Drazi uses the ring to get Piper and Dan, Prue and Jack, and oth...`

**Query ID:** `query_3`

**Document ID:** `doc_3`

**Document Length:** 50,157 characters

**Document Preview:**
```
OPEN IN LORELAI'S FRONT YARD
[An airport shuttle van drops Lorelai and Rory off in front of their house, then pulls away]
LORELAI: Agh!
RORY: And we're home.
LORELAI: How long does a freakin' van ride take?
RORY: Not that long!
LORELAI: Everybody in the world's life flashed before my eyes. That's how much time I had. I thought we were gonna die on that van.
RORY: It seemed a good possibility.
LORE...
```

**What to look for:**
- Query asks: When Cupid has his magic ring of love stolen by Drazi, the demon of hate, he turns to Phoebe for hel...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 5

**Query:** `When Magistrate Hale discovers that it was Isaac who broke the witches circle, it's up to John to save him. A power struggle arises within the hive, forcing Mary to assert her authority with potential...`

**Query ID:** `query_4`

**Document ID:** `doc_4`

**Document Length:** 40,982 characters

**Document Preview:**
```
[Scene: Paige's car. Paige is driving along the road, talking on her phone to Phoebe.]
Paige: Okay, so I've stopped at five herb shops but I finally found some eye of newt. So if it's good enough for Shakespeare's witches, I figured it'd help us put a serious dent in Cole.
Phoebe: Look, we've tried everything to vanquish him but nothing works, okay. So I just say we watch our backs and get on with...
```

**What to look for:**
- Query asks: When Magistrate Hale discovers that it was Isaac who broke the witches circle, it's up to John to sa...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 6

**Query:** `Amidst her sadness about Christopher, Lorelai has an intriguing dream about Luke; Rory returns from Washington to realize that she still may have feelings for Jess, but a chance encounter at the First...`

**Query ID:** `query_5`

**Document ID:** `doc_5`

**Document Length:** 17,891 characters

**Document Preview:**
```
New York is dangerous littered with thieves we've no morals here we just do as we please but I don't wanna go home where they all stare at me 'cause I'm tattooed and fired up and drunk and obscene. You wear your religion like a war sweater, you ask for the truth but you know you could do so much better and you sat on your fences and you screamed, "no retreat" so what will your legacy be?
AT CLOTHE...
```

**What to look for:**
- Query asks: Amidst her sadness about Christopher, Lorelai has an intriguing dream about Luke; Rory returns from ...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

---

#### Example 7

**Query:** `Logan offers Veronica a choice, but neither option is beneficial for their relationship. To salvage their friendship, Logan dumps Veronica as a girlfriend. Under pressure, Dean O'Dell reinstates the G...`

**Query ID:** `query_6`

**Document ID:** `doc_6`

**Document Length:** 8,768 characters

**Document Preview:**
```
Glenn: Lola, we have some good news and some bad news. The good news is, you don't have cancer.
Lola: Ohh!
Glenn: Cat just put nair in your shampoo.
Cat: Because you ate my lunch from the refrigerator.
Glenn: And the bad news is, she also put a chemical in your iced tea which turns your nose into a tennis ball. But it only lasts a second. So basically, everything's okay. Everything's okay.
Owen: C...
```

**What to look for:**
- Query asks: Logan offers Veronica a choice, but neither option is beneficial for their relationship. To salvage ...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content


---

