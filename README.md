# Report on GRPO reinforcement learning for a Viterbi algorithm with repetition suppression in non-autoregressive image captioning incorporating CRF.

## Introduction

Currently, autoregressive algorithms are used to generate text with generative AI. However, autoregressive algorithms require iterations corresponding to the number of generated tokens, thereby necessitating significant computation time. In contrast, non-autoregressive algorithms do not require iterations based on the word count, allowing for reduced computation time; however, n-gram repetitions tend to appear in the generated captions. The model addressed here for non-autoregressive image captioning consists of CLIP, an mlp_connector, BERT, and a CRF layer.

We proposed a Viterbi algorithm for this model that suppresses repetitions.

https://github.com/toshiouchi/stochastic_viterbi_sampling


We conducted GRPO reinforcement learning on an image captioning model based on the Viterbi algorithm with repetition suppression, and we report the results here.

## About the calculation


For training data, COCO train2017 was used with a split of train:val:dummy = 0.98:0.01:0.01. For testing, the 5,000 samples from val2017 were used as the test set. Using parameters obtained through pre-training,

https://github.com/toshiouchi/ImgCaptioning_LossCRF

, we performed reinforcement learning with GRPO samples generated via a Viterbi algorithm incorporating iterative suppression. Hyperparameters were determined using Optuna.

When calculating using the CRF loss based on the Viterbi algorithm with repetition suppression in the CRF layer, please set `self.use_crf_beam_logits = True` and `self.use_captions_beam = True`.

### Regarding Loss

The loss function accounts for policy_loss, kl_div_loss, entropy_loss, crf_loss, and ce_loss. The policy_loss is defined as clipped_ratio * advantages; the ratio is derived from sampled_log_probs, while the advantages are calculated from rewards. The crf_loss is based on the log-likelihood of a CRF layer that does not account for repetition suppression. The ce_loss represents the cross-entropy loss between the BERT emission probabilities and the ground-truth captions.

The reward took into account CIDEr, ROUGE-L, CLIP-Score, BERTScore, repetition penalty, length penalty, and unique n-gram ratio.

### On Approximation

In the CRF layer, we introduce a transition matrix representing transitions from sequence $i-1$ to sequence $i$. Since the transitions occur between tokens, the matrix has dimensions of `vocab_size` $\times$ `vocab_size`. Handling this matrix directly would require significant computational time and resources; therefore, we employed beam approximation (top-k approximation) and low-rank approximation.

## Calculation result

Measurements regarding the test data were taken before and after performing reinforcement learning.

befoer
```
CIDEr           0.801
rouge-L         0.504
clip_score      0.281
bert_score      0.835
repeat_count    0.00700
length_penalty -0.00129
```


`repeat_count` is the average sum of the repetition counts for 1-grams through 4-grams appearing in a single sentence. Regarding `length_penalty`, the closer the value is to 0, the closer the length of the generated caption is to that of the reference caption.

after
```
CIDEr           0.827
rouge-L         0.508
clip_score      0.281
bert_score      0.836
repeat_count    0.0700
length_penalty -0.00119
```


### Generated caption

before
```
hypo: [CLS] a woman filled with a blue arched tree living room. [SEP]
refe: [CLS] a room with chairs, a table, and a woman in it. [SEP]
hypo: [CLS] a black bear is standing in the grass. [SEP]
refe: [CLS] a big burly grizzly bear is show with grass in the background. [SEP]
hypo: [CLS] a room with a bedroom bookshel a bed. [SEP]
refe: [CLS] a bed room with a neatly made bed a window and a book shelf [SEP]
hypo: [CLS] a close up with a stop sign on it. [SEP]
refe: [CLS] an upside down stop sign by the road. [SEP]
hypo: [CLS] a group of stuffed animals next to each other. [SEP]
refe: [CLS] three stuffed animals are sitting on a bed. [SEP]
hypo: [CLS] a woman that skis on a snow covered slope. [SEP]
refe: [CLS] a young woman is skiing down the mountain slope. [SEP]
hypo: [CLS] a small has a kitchen with a stove. [SEP]
refe: [CLS] a white oven and a white refrigerator are in the kitchen. [SEP]
hypo: [CLS] two runner players catcher catch a baseball game. [SEP]
refe: [CLS] a couple of baseball player standing on a field. [SEP]
hypo: [CLS] a man swinging a tennis racquet. [SEP]
refe: [CLS] someone playing in a tennis tournament with a crowd looking on. [SEP]
hypo: [CLS] a group of a tennis patiently gettingets. [SEP]
refe: [CLS] the people are posing for a group photo. [SEP]
hypo: [CLS] a city gathers watching that are on the water. [SEP]
refe: [CLS] the woman is taking a photo of the white goose next to the river. [SEP]
hypo: [CLS] a woman holding a bluephone with a cell phone. [SEP]
refe: [CLS] a woman checking her cell phone with a hello kitty case. [SEP]
hypo: [CLS] a group of children wheels are on a train. [SEP]
refe: [CLS] a group of children ride on an indoor train. [SEP]
hypo: [CLS] a whiteffin meatw in a sandwich. [SEP]
refe: [CLS] a half eaten meal sitting on a plate. [SEP]
hypo: [CLS] a man on a surfboard in the water. [SEP]
refe: [CLS] a man with a wet suit on standing on a surfboard in the water. [SEP]
hypo: [CLS] a computer and a laptop with a desk. [SEP]
refe: [CLS] a picture of an imac desktop next to a mac laptop on a desk. [SEP]
hypo: [CLS] highway oil detour next to interstate signs. [SEP]
refe: [CLS] a street scene with focus on the street signs on an overpass. [SEP]
hypo: [CLS] a red double decker bus drives down the street. [SEP]
refe: [CLS] the red, double decker bus is driving past other buses. [SEP]
hypo: [CLS] a cat laying on top of a desk. [SEP]
refe: [CLS] a black and white cat relaxing inside a laptop. [SEP]
hypo: [CLS] a group of planes flying in the sky. [SEP]
refe: [CLS] a sky photo to jumbo jet airplanes over a bridge. [SEP]
hypo: [CLS] a baby iszzle next to each other zebra. [SEP]
refe: [CLS] a zebra in the grass who is cleaning himself. [SEP]
hypo: [CLS] a bed room with a neatly seat made. [SEP]
refe: [CLS] a simple modern bedroom sheets on the bed [SEP]
hypo: [CLS] a bus and white stopped at the street. [SEP]
refe: [CLS] city bus driving through pedestrian saturated area near crosswalk. [SEP]
hypo: [CLS] a black halves sitting on top of orange. [SEP]
refe: [CLS] a white bowl of green granny smith apples. [SEP]
hypo: [CLS] a man player is batter to a baseball game. [SEP]
refe: [CLS] a baseball player swinging a bat over home plate. [SEP]
hypo: [CLS] a plate topped with lots of a tableeti it. [SEP]
refe: [CLS] a nicely set dining table filled with food and a cake topped with berries. [SEP]
hypo: [CLS] a boy in a surfboard in the ocean. [SEP]
refe: [CLS] a man in a black shirt plays on an ocean wave. [SEP]
hypo: [CLS] a group and children suits photo of a large class [SEP]
refe: [CLS] a group of children standing and sitting beside each other. [SEP]
hypo: [CLS] a close up topped with a plate on it. [SEP]
refe: [CLS] a meal of burnt toast slices with condiments on the side. [SEP]
hypo: [CLS] a snowboarder is jumping on the air. [SEP]
refe: [CLS] a snow skier is doing an aerial trick [SEP]
hypo: [CLS] a person cross country skiing on a hill. [SEP]
refe: [CLS] a person is skiing on a snowy hill top. [SEP]
hypo: [CLS] a banana pastry sitting on top of a table. [SEP]
refe: [CLS] a banana and donut in the same plastic bag. [SEP]
hypo: [CLS] a cup with a sitting on a pair of a table. [SEP]
refe: [CLS] there is a white coffee cup with a skull and bones on it next to a knife [SEP]
hypo: [CLS] a group of people standing in a wine. [SEP]
refe: [CLS] a couple of people are standing in front of some wine bottles [SEP]
hypo: [CLS] a group of being birds near a field. [SEP]
refe: [CLS] ships in the distance behind a grassy marsh with egrets [SEP]
hypo: [CLS] a man sitting on top of in a toilet. [SEP]
refe: [CLS] a young man bending next to a toilet. [SEP]
hypo: [CLS] a skier walking people on top of a hill. [SEP]
refe: [CLS] a group of people riding skis on a snowy surface [SEP]
hypo: [CLS] a white plate of rice spoon and broccoli. [SEP]
refe: [CLS] a plate of broccoli, rice, meat and other vegetables [SEP]
hypo: [CLS] a person ridinger is board on a skateboard [SEP]
refe: [CLS] a man riding a skateboard on piece of concrete in a park. [SEP]
hypo: [CLS] a bunch of a next to each other bananas. [SEP]
refe: [CLS] a bunch of bananas very close up on a table. [SEP]
hypo: [CLS] a white plate of fried meat and broccoli. [SEP]
refe: [CLS] a dinner plate that has white steamed rice with stir fry vegetables and chicken. [SEP]
hypo: [CLS] a girl child throwing wi asian playing living room. [SEP]
refe: [CLS] a little girl holding a white nintendo wii game controller. [SEP]
hypo: [CLS] two men standing next to each other gentleman. [SEP]
refe: [CLS] a man is shaking hands with another man. [SEP]
hypo: [CLS] a man in a white shirt posing for tie. [SEP]
refe: [CLS] a man in a suit poses for the camera. [SEP]
hypo: [CLS] a living room with a couch next television. [SEP]
refe: [CLS] a corner of a living room with a tv in it. [SEP]
hypo: [CLS] a person riding a surfboard on the ocean. [SEP]
refe: [CLS] a man riding a surfboard on a wave in the ocean. [SEP]
hypo: [CLS] a cat sitting on top of a laptop computer keyboard. [SEP]
refe: [CLS] gray cat looking at electronic monitor next to phone. [SEP]
hypo: [CLS] a few signing teaching standing in front of people. [SEP]
refe: [CLS] a group of people cutting a ribbon on a street. [SEP]
hypo: [CLS] a bus and white stopped at the street. [SEP]
refe: [CLS] a bus is traveling down the city street. [SEP]
hypo: [CLS] a man standing in front of a mirror. [SEP]
refe: [CLS] a man sitting cross legged in front of a mirror on the floor taking a self portrait [SEP]
hypo: [CLS] a group of people standing next to each other. [SEP]
refe: [CLS] boys posing for a picture beside two surf boards. [SEP]
hypo: [CLS] a large plane is sitting on a runway. [SEP]
refe: [CLS] a brown lot airliner sitting on the tarmac [SEP]
hypo: [CLS] a dirty diggingder next to a toilet. [SEP]
refe: [CLS] a person relieving themselves in a white toilet. [SEP]
hypo: [CLS] a skier riding skis on a hill. [SEP]
refe: [CLS] a person on skis skiing down a mountain slope. [SEP]
hypo: [CLS] a man player is playing tennis racket. [SEP]
refe: [CLS] a man playing tennis hits a ball across the court [SEP]
hypo: [CLS] a white plate topped with a bowlstu. [SEP]
refe: [CLS] a plate has two bowls on it with two different types of food, one looks like pickled onions and the other looks like cooked meat. [SEP]
hypo: [CLS] a herd of sheep standing next to each other. [SEP]
refe: [CLS] group of sheep standing next to a brick building. [SEP]
hypo: [CLS] a bunch of orange next to each other bananas. [SEP]
refe: [CLS] a bunch of bananas with an apple sitting in the middle. [SEP]
hypo: [CLS] a train that is traveling down a track. [SEP]
refe: [CLS] a close - up of an electric train going by on the tracks. [SEP]
hypo: [CLS] a bathroom with a bath, shower sink. [SEP]
refe: [CLS] a bathroom view of a tub and sink wit mirrors [SEP]
hypo: [CLS] a man on a surfboard in the ocean. [SEP]
refe: [CLS] a person riding a surf board on a wave [SEP]
hypo: [CLS] a batter player swinging a bat at a baseball game. [SEP]
refe: [CLS] a hitter is waiting for the pitch to be thrown [SEP]
hypo: [CLS] a assorted grapes types of orange fruits. [SEP]
refe: [CLS] a black and white photo of nuts and fruit [SEP]
hypo: [CLS] a bridge drives filled with a city street. [SEP]
refe: [CLS] a building lined street with three lanes and light traffic. [SEP]```
```
after
```
hypo: [CLS] a woman filled with a blue arched tree living room. [SEP]
refe: [CLS] a room with chairs, a table, and a woman in it. [SEP]
hypo: [CLS] a black bear is standing in the grass. [SEP]
refe: [CLS] a large bear that is sitting on grass. [SEP]
hypo: [CLS] a room with a bedroom bookshel a bed. [SEP]
refe: [CLS] this room has a bed with blue sheets and a large bookcase [SEP]
hypo: [CLS] a close up with a stop sign on it. [SEP]
refe: [CLS] a stop sign put upside down on a metal pole [SEP]
hypo: [CLS] a group of stuffed animals next to each other. [SEP]
refe: [CLS] a group of three stuffed animal teddy bears. [SEP]
hypo: [CLS] a woman that skis on a snow covered slope. [SEP]
refe: [CLS] a person on skis makes her way through the snow [SEP]
hypo: [CLS] a small has a kitchen with a stove. [SEP]
refe: [CLS] kitchen appliances and cabinets as seen through opening. [SEP]
hypo: [CLS] two runner players catcher catch a baseball game. [SEP]
refe: [CLS] two baseball players are playing baseball on a field [SEP]
hypo: [CLS] a man swinging a tennis racquet. [SEP]
refe: [CLS] a male tennis player in white shorts is playing tennis [SEP]
hypo: [CLS] a group of a tennis patiently gettingets. [SEP]
refe: [CLS] the people are posing for a group photo. [SEP]
hypo: [CLS] a city gathers watching that are on the water. [SEP]
refe: [CLS] a beautiful woman taking a picture with her smart phone. [SEP]
hypo: [CLS] a woman holding a bluephone with a cell phone. [SEP]
refe: [CLS] a woman in white shirt holding up a cellphone. [SEP]
hypo: [CLS] a group of children wheels are on a train. [SEP]
refe: [CLS] several children on a small indoor kiddie train. [SEP]
hypo: [CLS] a whiteffin meatw in a sandwich. [SEP]
refe: [CLS] a plate with a burger that is halfway eaten. [SEP]
hypo: [CLS] a man on a surfboard in the water. [SEP]
refe: [CLS] a man in a wet suit stands on a surfboard and rows with a paddle. [SEP]
hypo: [CLS] a computer and a laptop with a desk. [SEP]
refe: [CLS] a picture of an imac desktop next to a mac laptop on a desk. [SEP]
hypo: [CLS] highway oil detour next to interstate signs. [SEP]
refe: [CLS] some cars on the freeway are exiting onto sunset blvd. [SEP]
hypo: [CLS] a red double decker bus drives down the street. [SEP]
refe: [CLS] the red, double decker bus is driving past other buses. [SEP]
hypo: [CLS] a cat laying on top of a desk. [SEP]
refe: [CLS] a black fluffy cat sitting on top of a computer keyboard. [SEP]
hypo: [CLS] a group of planes flying in the sky. [SEP]
refe: [CLS] two airplanes flying in the sky above a black bridge. [SEP]
hypo: [CLS] a baby iszzle next to each other zebra. [SEP]
refe: [CLS] a baby giraffe drinking milk from it ' s mother in a field. [SEP]
hypo: [CLS] a bed room with a neatly seat made. [SEP]
refe: [CLS] a simple modern bedroom sheets on the bed [SEP]
hypo: [CLS] a bus and white stopped at the street. [SEP]
refe: [CLS] a big purple bus parked in a parking spot [SEP]
hypo: [CLS] a black halves sitting on top of orange. [SEP]
refe: [CLS] a white bowl of green granny smith apples. [SEP]
hypo: [CLS] a man player is batter to a baseball game. [SEP]
refe: [CLS] a baseball player swinging a bat over home plate. [SEP]
hypo: [CLS] a plate topped with lots of a tableeti it. [SEP]
refe: [CLS] a table topped with a cake covered in berries next to a plate of sandwiches. [SEP]
hypo: [CLS] a boy in a surfboard in the ocean. [SEP]
refe: [CLS] the boy is looking back at a wave in the ocean. [SEP]
hypo: [CLS] a group and children suits photo of a large class [SEP]
refe: [CLS] a black and white photo of a group of kids. [SEP]
hypo: [CLS] a close up topped with a plate on it. [SEP]
refe: [CLS] a meal of burnt toast slices with condiments on the side. [SEP]
hypo: [CLS] a snowboarder is jumping on the air. [SEP]
refe: [CLS] a snow skier is doing an aerial trick [SEP]
hypo: [CLS] a person cross country skiing on a hill. [SEP]
refe: [CLS] a skier stands on skis at the top of a snowy plateau. [SEP]
hypo: [CLS] a banana pastry sitting on top of a table. [SEP]
refe: [CLS] a banana and donut in the same plastic bag. [SEP]
hypo: [CLS] a cup with a sitting on a pair of a table. [SEP]
refe: [CLS] a close up of a knife and a cup on a surface [SEP]
hypo: [CLS] a group of people standing in a wine. [SEP]
refe: [CLS] a group of people standing around each other. [SEP]
hypo: [CLS] a group of being birds near a field. [SEP]
refe: [CLS] a couple of birds walk through a field of tall grass near a sail boat harbor. [SEP]
hypo: [CLS] a man sitting on top of in a toilet. [SEP]
refe: [CLS] a man is kneeling and holding on to a toilet. [SEP]
hypo: [CLS] a skier walking people on top of a hill. [SEP]
refe: [CLS] the paths of some snowboarders carved in a mountain slope. [SEP]
hypo: [CLS] a white plate of rice spoon and broccoli. [SEP]
refe: [CLS] a plate of broccoli, rice, meat and other vegetables [SEP]
hypo: [CLS] a person ridinger is board on a skateboard [SEP]
refe: [CLS] a skateboarder is riding on boards that have been placed over grass. [SEP]
hypo: [CLS] a bunch of a next to each other bananas. [SEP]
refe: [CLS] a bunch of bananas sitting on top of a wooden table. [SEP]
hypo: [CLS] a white plate of fried meat and broccoli. [SEP]
refe: [CLS] a plate of food and a drink on a table. [SEP]
hypo: [CLS] a girl child throwing wi asian playing living room. [SEP]
refe: [CLS] a young girl playing a video game while others talk. [SEP]
hypo: [CLS] two men standing next to each other gentleman. [SEP]
refe: [CLS] two men shaking hands after a dinner speech. [SEP]
hypo: [CLS] a man in a white shirt posing for tie. [SEP]
refe: [CLS] a man in a shirt and tie motioning with his hand. [SEP]
hypo: [CLS] a living room with a couch next television. [SEP]
refe: [CLS] a television, couch and chair in the corner of a room. [SEP]
hypo: [CLS] a person riding a surfboard on the ocean. [SEP]
refe: [CLS] a surfer is riding on a small wave. [SEP]
hypo: [CLS] a cat sitting on top of a laptop computer keyboard. [SEP]
refe: [CLS] a cat is sitting and watching a computer screen. [SEP]
hypo: [CLS] a few signing teaching standing in front of people. [SEP]
refe: [CLS] a man cutting a ribbon at a ceremony [SEP]
hypo: [CLS] a bus and white stopped at the street. [SEP]
refe: [CLS] a bus sitting still on the side of the road. [SEP]
hypo: [CLS] a man standing in front of a mirror. [SEP]
refe: [CLS] a man sitting cross legged in front of a mirror on the floor taking a self portrait [SEP]
hypo: [CLS] a group of people standing next to each other. [SEP]
refe: [CLS] a group of people in room with surfboards. [SEP]
hypo: [CLS] a large plane is sitting on a runway. [SEP]
refe: [CLS] large mustard yellow commercial airplane parked in the airport [SEP]
hypo: [CLS] a dirty diggingder next to a toilet. [SEP]
refe: [CLS] a photo of a duvet style toilet shooting water. [SEP]
hypo: [CLS] a skier riding skis on a hill. [SEP]
refe: [CLS] a skier is in the snow going downhill. [SEP]
hypo: [CLS] a man player is playing tennis racket. [SEP]
refe: [CLS] a man playing tennis hits a ball across the court [SEP]
hypo: [CLS] a white plate topped with a bowlstu. [SEP]
refe: [CLS] a close up of two bowls of food on a table [SEP]
hypo: [CLS] a herd of sheep standing next to each other. [SEP]
refe: [CLS] sheep standing around next to a brick wall [SEP]
hypo: [CLS] a bunch of orange next to each other bananas. [SEP]
refe: [CLS] a bunch of bananas with an apple sitting in the middle. [SEP]
hypo: [CLS] a train that is traveling down a track. [SEP]
refe: [CLS] a blue and white train is moving on the rails. [SEP]
hypo: [CLS] a bathroom with a bath, shower sink. [SEP]
refe: [CLS] a shower curtain sits open in an empty and clean bathroom. [SEP]
hypo: [CLS] a man on a surfboard in the ocean. [SEP]
refe: [CLS] a man is surfing and a wave is crashing [SEP]
hypo: [CLS] a batter player swinging a bat at a baseball game. [SEP]
refe: [CLS] a baseball player holds his bat and waits for the pitch. [SEP]
hypo: [CLS] a assorted grapes types of orange fruits. [SEP]
refe: [CLS] a black and white photo of nuts and fruit [SEP]
hypo: [CLS] a bridge drives filled with a city street. [SEP]
refe: [CLS] a street with a row of older red brick building on one side. [SEP]
```

## Summary


All scores improved after reinforcement learning, with the exception of clip_score and repeat_count; clip_score and repeat_count remained unchanged.

## Points to note regarding the program.

1. A custom-developed stochastic Viterbi sampling function for GRPO is used; this is based on the previously shared URL.
1. The image captioning model performs calculations three times: for sampling, training, and KL divergence reference. Top-k approximation is applied in each case. The `torch.topk` function is executed only during sampling; for training and reference calculations, `beam_emission_scores` are computed using the `top_indices` obtained during sampling.
1. The token paths used to derive `sampled_log_probs` from `log_probs` are also entirely based on the token paths from the sampling phase.
1. When a CRF is employed, the token at sequence $i-1$ (pre-transition) must be specified to calculate `log_probs`. During sampling, the token paths determined by the sampling process were applied to both sequence $i$ (post-transition) and sequence $i-1$ (pre-transition). Similarly, during training and calculations for the KL divergence reference model, the token paths obtained during sampling were applied to both sequence $i$ and sequence $i-1$.



