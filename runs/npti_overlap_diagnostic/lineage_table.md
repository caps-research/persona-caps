# Direct lineage diagnostic (same-trait, same-polarity)

| Pair | n | cos top-1 mean | cos top-5 mean | trigram-max mean |
|---|---|---|---|---|
| P2_to_IPIP | 800 | 0.499 | 0.430 | 0.005 |
| CAPS_to_IPIP | 166 | 0.441 | 0.379 | 0.001 |
| CAPS_to_P2 | 166 | 0.503 | 0.480 | 0.001 |
| IPIP_to_P2 | 300 | 0.371 | 0.347 | 0.002 |

## Top-5 P^2 -> IPIP matches (closest paraphrases)

**1. trait=neuroticism pol=high cos=0.657**
- P^2 (p2_neuroticism_high_32): You often find yourself dealing with anxiety and stress, making it challenging to relax and feel at ease. Your emotions can be intense, and you may struggle with self-doubt and negative thoughts. It's common for you to feel overwhelmed by l
- IPIP (ipip_146): Get overwhelmed by emotions.

**2. trait=neuroticism pol=high cos=0.654**
- P^2 (p2_neuroticism_high_41): You often feel overwhelmed by stress and anxiety, finding it difficult to relax and unwind. Your emotions can be intense and hard to manage, leading to frequent feelings of self-doubt and insecurity. You are highly sensitive to criticism an
- IPIP (ipip_146): Get overwhelmed by emotions.

**3. trait=neuroticism pol=high cos=0.636**
- P^2 (p2_neuroticism_high_44): You often experience strong emotional reactions and find it challenging to manage stress. You are prone to feelings of anxiety and worry about various aspects of your life. You may struggle with self-doubt and find it difficult to relax. Yo
- IPIP (ipip_146): Get overwhelmed by emotions.

**4. trait=neuroticism pol=high cos=0.631**
- P^2 (p2_neuroticism_high_8): You often find yourself experiencing a range of intense emotions. You are prone to anxiety and can be quite self-critical, frequently worrying about various aspects of your life. Your emotional sensitivity means you are very attuned to your
- IPIP (ipip_146): Get overwhelmed by emotions.

**5. trait=neuroticism pol=high cos=0.630**
- P^2 (p2_neuroticism_high_19): You often find yourself experiencing a range of intense emotions, from anxiety to sadness. You tend to worry about potential problems and may struggle with self-doubt. Your sensitivity to stress can sometimes lead to mood swings, making it 
- IPIP (ipip_131): Have frequent mood swings.

## Top-5 CAPS -> IPIP matches (closest)

**1. trait=neuroticism pol=low cos=0.695**
- CAPS (beck2022_low_neur__positivity__tense__143): When I'm in a pleasant situation with good company I stay relaxed and at ease, my breathing is even, my shoulders are loose, I smile and engage comfortably
- IPIP (ipip_181): Am relaxed most of the time.

**2. trait=neuroticism pol=high cos=0.620**
- CAPS (beck2022_high_neur__lonely__temperamental__150): When I notice I'm alone and have no one to reach out to I become completely overwhelmed by intense emotions, my chest tightens, tears spill out, and I can't regain control
- IPIP (ipip_146): Get overwhelmed by emotions.

**3. trait=neuroticism pol=high cos=0.614**
- CAPS (beck2022_high_neur__lonely__controls_emotions__157): When isolation and loneliness wash over me My emotions burst out uncontrollably, I lose the ability to keep my feelings in check and feel completely overwhelmed
- IPIP (ipip_146): Get overwhelmed by emotions.

**4. trait=extraversion pol=high cos=0.601**
- CAPS (beck2022_high_extr__positivity__others_charge__073): When I find myself in a pleasant, comfortable setting with people around I take charge of the conversation, lead what we do next, and direct the energy of the group rather than letting others steer
- IPIP (ipip_67): Enjoy being part of a group.

**5. trait=extraversion pol=high cos=0.567**
- CAPS (beck2022_high_extr__excited_schoolwork__shy__089): When I'm excited about a school assignment or project I raise my hand, speak out, and jump into group discussions without waiting
- IPIP (ipip_67): Enjoy being part of a group.
