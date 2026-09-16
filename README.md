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

# The following section reports the results of performing the same procedures as described above on the v7 dataset.

While caption generation on the COCO dataset proceeded smoothly (with a CIDEr score of around 0.8) and was relatively resistant to reward hacking during reinforcement learning, caption generation on the v7 dataset proved more challenging (with a CIDEr score of around 0.2) and was plagued by reward hacking issues; however, in the current report, no reward hacking appears to have occurred.

## Calculation result

Measurements regarding the test data were taken before and after performing reinforcement learning.

befoer
```
CIDEr           0.197
rouge-L         0.538
clip_score      0.272
bert_score      0.845
repeat_count    0.325
length_penalty -0.0686
```
`repeat_count` is the average sum of the repetition counts for 1-grams through 4-grams appearing in a single sentence. Regarding `length_penalty`, the closer the value is to 0, the closer the length of the generated caption is to that of the reference caption.

after
```
CIDEr           0.211
rouge-L         0.542
clip_score      0.273
bert_score      0.847
repeat_count    0.328
length_penalty -0.0681
```

## Generated Captions

before
```
hypo: [CLS] in this image we can see a group of people. in the ground. there are standing. on the floor. [SEP]
refe: [CLS] in this image there are some persons standing at right side of this image and there are some lights are arranged at top of this image. there are some cabins as we can see at left side of this image and there is a floor at bottom of this image. [SEP]
hypo: [CLS] in this picture we can see a utens flame oven. [SEP]
refe: [CLS] in this image we can see a pie on a large spatula. on the backside we can see some fire. [SEP]
hypo: [CLS] this picture describes about are a room. in the image, there is a cake. on the background, we can see a table. [SEP]
refe: [CLS] in this image, we can see a table, on that table there is a cake and we can see a candle on the cake, there is a black color laptop on the table, in the background we can see a wall and there are some photos on the wall. [SEP]
hypo: [CLS] in this picture we can see there are placed on the floor. [SEP]
refe: [CLS] in the image we can see a t - shirt, black in color, on it there is a text. we can even see there are many flower bookies. this is a footpath, brick wall and fruits. [SEP]
hypo: [CLS] in this image there are observe a group of people standing on the floor. in the background we can see the ground. [SEP]
refe: [CLS] in this image in the foreground there are two persons one person is wearing a hat, and in the background there are group of people one person is holding a pole and flag and some boards poles, buildings, mountains. at the bottom there are some stones, and at the top there is sky. [SEP]
hypo: [CLS] in this picture we can see a machine. [SEP]
refe: [CLS] in this picture we can see a wheel, a gear and springs in the front, in the background there is grass. [SEP]
hypo: [CLS] in this picture we can see a person standing on the floor. in her hand. in the ground. [SEP]
refe: [CLS] here we can see a woman having a football in her hand and behind her we can see group of people standing and it is snowy [SEP]
hypo: [CLS] this image a picture are a room. on the right side of a laptop, we can see a bottle, there is a table. [SEP]
refe: [CLS] in this image there is a bed, on the bed there are the pillows and beside the bed there is a table, on the table there is a laptop and there is a fish pot, under the table there is a bottle kept on the table and there is a speaker on the floor. [SEP]
hypo: [CLS] this picture is clicked outside buildings. at the ground. in the road, there are trees. [SEP]
refe: [CLS] in this picture we can see vehicles on the road, trees, poles and in the background we can see the sky with clouds. [SEP]
hypo: [CLS] in this picture we can see a building. in the ground. in the image. in the right side there are buildings. on the road. [SEP]
refe: [CLS] in this picture, we can see a few buildings, poles, stairs, railing, road, a few people, and vehicles, and the sky with clouds, and we can see date and time in bottom right side of the picture. [SEP]
hypo: [CLS] this inside a picture is a room. on the image, we can see a table, there are some other objects. [SEP]
refe: [CLS] this image is taken in a store where we can see mirrors, chairs, wall, lights, ceiling, bottles, tripod stands, frames, jar, table and a cupboard. [SEP]
hypo: [CLS] this picture describes about are observe a room. here we can see a chair, there is a table. [SEP]
refe: [CLS] this is a picture of a kitchen. in this picture there are chairs, table, microwave oven, lights closet, mat and many other kitchen utensils. on the left there is a refrigerator, in the refrigerator there are drinks and other food items. [SEP]
hypo: [CLS] in front a person is a man standing in the ground. in the image there are walking on the background we can see a building. [SEP]
refe: [CLS] in this image there is a person walking, plants, grass, bench, buildings, pole, antenna, trees, sky. [SEP]
hypo: [CLS] in this picture we can see a car on the ground. in the road. [SEP]
refe: [CLS] in this picture we can see car on the road and we can see grass and helmet. in the background of the image it is blurry. [SEP]
hypo: [CLS] in this image we can see a black which are sitting on the background there is a tree. [SEP]
refe: [CLS] in this image i can see an animal which is in white color. i can see the pole. in the background i can see many trees and the sky. [SEP]
hypo: [CLS] in this picture we can see a group of people standing and they are smiling. in their hands. [SEP]
refe: [CLS] in this image we can see people, rods and wall. among them two people are holding bottles and one woman wore handbag. [SEP]
hypo: [CLS] in this picture we can see a building. in the image. in the right side, buildings. in the sky. there are trees. on the road. [SEP]
refe: [CLS] in the middle of the image we can see some buildings, trees, plants and two persons walking on the road. at the top of the image there are some clouds in the sky. [SEP]
hypo: [CLS] in the foreground of a a person wearing goggles and she is standing and gloves, we can see the background there are trees. [SEP]
refe: [CLS] the woman in blue t - shirt and black trouser is highlighted. she wore gloves, goggles. far there are number of bare trees. snow is in white color. [SEP]
hypo: [CLS] in this image we can see a cat in the floor. on the background there is a wall. [SEP]
refe: [CLS] in this image we can see an animal on the ground. in the background, we can see the wall with ventilation. [SEP]
hypo: [CLS] in this image there are a woman who is a few people standing on the right side of the background we can see a screen. [SEP]
refe: [CLS] in this picture we can see a person wearing a black object on the head. we can see some text on the black object. there is a phone visible in the hands of a person in the bottom right. we can see screen a few people and other objects in the background. background is blurry. [SEP]
hypo: [CLS] in the bottom it is observe a group of people standing on the image we can see the background there are trees. [SEP]
refe: [CLS] in the picture i can see people among them a person is standing behind the wooden fence and the people in front of the image are wearing helmets and some other objects. in the background i can see trees, an umbrella and some other objects. [SEP]
hypo: [CLS] in this picture we can see a house in the grass. in the ground and plants and there are trees. [SEP]
refe: [CLS] in this image there is an old house and some algae on the house, around the house there are trees, plants and some dry leaves on the surface. [SEP]
hypo: [CLS] in this picture we can see a building. in the image. and some buildings. in the ground. there are trees. on the road. [SEP]
refe: [CLS] in this image we can see buildings, staircases, railings, sign boards, motor vehicles on the road, persons walking on the floor, electric cables, trees and sky with clouds. [SEP]
hypo: [CLS] in this picture we can see a man riding the cycle on the background there is a bicycle. [SEP]
refe: [CLS] a man is riding a bicycle. man wears blue vest and blue denim pant with casual shoe. he has a long beard and short hair, wears spectacles. there is a wall and a pillar to left of him. from a distance, it is looking as if he is passing through a bridge. [SEP]
hypo: [CLS] in this image there is a painting. here we can see a woman sitting on the swing. [SEP]
refe: [CLS] in this image we can see there is a poster with a painting. and there is a person sitting on the object. there is a pillow, umbrella, cloth, flowers and tree. [SEP]
hypo: [CLS] in this picture, we observe a man standing in the image, there is a boy sitting on the background we can see the ground. [SEP]
refe: [CLS] in this picture i can see a woman and a boy standing. i can see buildings, trees, hills and a blue cloudy sky. i can see grass on the ground. [SEP]
hypo: [CLS] in this picture we can see a building. [SEP]
refe: [CLS] in this image i can see buildings and windows. this image is taken may be during a day. [SEP]
hypo: [CLS] in this image we can see a butterfly in the leaf. on the background there is blurred. [SEP]
refe: [CLS] in this image we can see a butterfly is sitting on white and yellow color flower. behind so many flowers are there. [SEP]
hypo: [CLS] in this image there is observe a person wearing clothes and holding an object. in front of the background we can see some objects. [SEP]
refe: [CLS] in this picture there is a woman wearing white color t - shirt with cap is holding a rod in the grill burner. behind you can see white and blue color burner. in the background there is a white cladding tiles and some plates placed on the top. [SEP]
hypo: [CLS] in this image, we observe a rat which is a tree, we can see a plant. [SEP]
refe: [CLS] in this picture we can see a person ' s fingers and in the background we can see a wooden object, mice, plants with flowers and wooden sticks. [SEP]
hypo: [CLS] in this picture, we observe a woman in the image there is standing on the background we can see a wall. [SEP]
refe: [CLS] in the middle of the image a woman is standing and smiling. beside her we can see a painting on a wooden board. behind her we can see a wall. [SEP]
hypo: [CLS] in the foreground of a woman with a girl is standing on the ground, we can see the background, plants, there are trees. [SEP]
refe: [CLS] in this picture we can see a girl, she is standing in front of the plants, in the background we can see a room and few trees. [SEP]
hypo: [CLS] in this image, we can see a few people standing on the ground. in front of them there are some objects. [SEP]
refe: [CLS] in this picture i can see group of people, there are baskets, those are looking like sweet potatoes, and in the background there are trees. [SEP]
hypo: [CLS] this picture describes about are wearing a man sitting in the chairs, we can see the chair. in front of him, there is a microphone. on the table. [SEP]
refe: [CLS] this image is taken indoors. in the background there is a wall with a door and there is a person standing on the floor. at the bottom of the image there is a table with a tablecloth, a name board and a few things on it. on the right side of the image a woman and a person are sitting on the chairs and a man is standing on the floor. in the middle of the image a man is sitting on the chair. on the left side of
hypo: [CLS] this image a picture is a room. in the chairs. in the chair, there are glasses. in front of them. in the background. on the table. [SEP]
refe: [CLS] in this picture there are three men who are sitting on the chair. there is a book. there is a glass. there is a spectacle on the table. there is a frame on the wall. there is a poster. there is a door and there is a device. [SEP]
hypo: [CLS] in this picture we can see trees, which is a christmas tree. [SEP]
refe: [CLS] in this image in the center there is a tree and some lights and decorations, and there is black background. [SEP]
hypo: [CLS] in this image we can see a poster. [SEP]
refe: [CLS] this image consists of a poster and this is a black and white image. on the left side a person is wearing a jacket, standing and looking at the picture. on the right side, i can see some text. [SEP]
hypo: [CLS] in this picture we can see a house. on the image there is a building. [SEP]
refe: [CLS] this is the picture of a shed. in this image there is a person walking. there is a text and painting on the wall. in the foreground there are pipes. at the back there are trees and poles. at the top there is sky. [SEP]
hypo: [CLS] in this picture we can see a bottle on the table. [SEP]
refe: [CLS] this is a picture, in this picture there is a green color bottle and a wine glass on a wooden table. background of this bottle and a glass is a wall which is in white color. [SEP]
hypo: [CLS] in this picture we can see two persons and a man standing on the floor. in front of them there is a table. [SEP]
refe: [CLS] in this image i can see a woman sitting on chair and 2 person standing behind her. i can see a table on which i can see a jar and 2 plates. [SEP]
hypo: [CLS] this are a black and white image there is a woman sitting on the background. [SEP]
refe: [CLS] in this picture there is a man who is wearing t - shirt and short. she is holding a flower. in the back i can see the darkness. [SEP]
hypo: [CLS] in front a man is observe a group of people and holding an object. in the image there are standing on the background we can see some other objects. [SEP]
refe: [CLS] in this image there are people. there is a table on the right side and there are glasses. there is a cloth at the top. there are trees. there is sky. [SEP]
hypo: [CLS] in this image there are observe a group of people standing on the background we can see an me is holding an object. [SEP]
refe: [CLS] in this image, we can see people and some are wearing uniforms and holding boards with some text and one of them is wearing a hat. in the background, there are some objects and we can see a cardboard and there is a wall. [SEP]
hypo: [CLS] in this image men are two persons standing. in front of the background there is a microphone. [SEP]
refe: [CLS] in this image i can see on the left side a man is holding the microphone, he wore black color coat, on the right side there is another man. he wore green color coat, white color cap, behind them there is the white color cloth to an iron grill. [SEP]
hypo: [CLS] in this image clicked is observe a group of people standing on the backside we can see the background there are trees. [SEP]
refe: [CLS] in the middle a man is there, he wore white color cap and a garland. here a beautiful woman is standing, she wore orange color dress beside him a man is there, he wore t - shirt, goggles. there are trees at the backside of an image. on the right side there is a house. [SEP]
hypo: [CLS] in this image there is observe a screen, we can see some text. [SEP]
refe: [CLS] this is an edited image. in this image, in the middle, we can see a mobile, in the mobile, we can see some pictures and a keyboard. on the right side, we can see some text. [SEP]
hypo: [CLS] in this image there are many fruits. [SEP]
refe: [CLS] in this picture we can see fruits in the water. [SEP]
hypo: [CLS] in this image there is observe a man standing in the ground. on the background we can see a dog. [SEP]
refe: [CLS] in this picture there are three persons one, two, three, one is a lady among them, a lady is holding the baby and the person who is standing at the right side of the image it seems to be walking by holding the leach of the dog and the person who is standing in the middle of the image is watching in front of the direction and the area where they were stood it seems to be greenery and there is a building at the right side of the image on
hypo: [CLS] this picture an edited image is a woman standing in front of the background we can see a christmas tree. [SEP]
refe: [CLS] this picture is an animated image. we can see a lady is standing on the left side of the image. on the right side of the image a christmas tree is there. in the background of the image we can see a wall. at the bottom of the image floor is there. [SEP]
hypo: [CLS] in the foreground of a truck. in the ground. these are a railway track. i can see the background there is a train. on the road. [SEP]
refe: [CLS] in the picture we can see a street, in the street we can see a train on the track which is green in color with some yellow lines and besides, we can see some people standing on the path near to the buildings and we can also see some poles, street lights, wires to the poles and sky with a cloud. [SEP]
hypo: [CLS] in this image, we can see a few people standing on the floor. in front of them there are some objects. [SEP]
refe: [CLS] in the image few people are standing and sitting and smiling and there are some wires and there is a chair. they are in a wooden building and there is a window. through the window we can see some trees and there is a vehicle. [SEP]
hypo: [CLS] in this picture we can see a plant. [SEP]
refe: [CLS] in this image we can see flowers and buds. in the top right, we can see a rock. the background of the image is blurred. [SEP]
hypo: [CLS] in this picture we can see a sign board on the image there is a tree. [SEP]
refe: [CLS] in the center of the image we can see a sign board attached to the pole. in the background we can see the tree. [SEP]
hypo: [CLS] in this image there is a moving on. in the tracks. in the railway track. on the background i can see a train. [SEP]
refe: [CLS] in this image there is a metro train under the metro train there is a house, near the house there is a road on that road there are light poles and signal poles, cars are parked to a side, in the background there is a big building. [SEP]
hypo: [CLS] in the a black and white image which is a statue. in the background there are lights. [SEP]
refe: [CLS] in this image we can see sculptures and a wall. behind the sculptures we can see lamps and water. the background of the image is dark. [SEP]
hypo: [CLS] this picture describes about are observe a man standing on the right side, there is a microphones, we can see the stage. [SEP]
refe: [CLS] in this picture, we can see a few people, we can see the stage, and some objects on the stage like microphones with stands, speakers, and in the background we can see the screen. [SEP]
hypo: [CLS] in this image there is observe a group of people standing on the background we can see a wall. [SEP]
refe: [CLS] this is an edited picture. i can see three persons standing, there is a person holding a camera, and in the background there is a door and a board. [SEP]
hypo: [CLS] in this picture of the image there are toys, we can see the floor. [SEP]
refe: [CLS] in this picture, we see the toys which are holding some electronic goods. we see these toys are placed on the white table or thermocol. beside that, we see an objects in black color which looks like the solar panels. on either side of the panels, we see the walls, which are made up of thermocol. we even see the railing and this wall is in blue color. in the background, we see a building in white color. this
hypo: [CLS] in this image, we observe a girl is standing on the ground, we can see the background there are trees. [SEP]
refe: [CLS] in this image i can see a person standing. she is wearing green, black and white dress and black shoe. background i can see trees and wall. [SEP]
hypo: [CLS] this are a black and white picture. in the image there is a man standing on the background we can see a door. [SEP]
refe: [CLS] in this picture there is a man standing on the floor and carrying a bag and we can see glass, through glass we can see car on the road, rods, people and buildings. in the background of the image we can see a board on the wall. [SEP]
hypo: [CLS] in this picture, we can see a few people sitting in the floor. in front of them there are some objects. on the table. [SEP]
refe: [CLS] in the center of the image there are four people sitting around a dining table behind them there are three people standing. in the background there is a door and a curtain. on the left side of the room there is a bin. on the right side there is a cupboard. there are glasses, cups, spoons and plates placed on the table. [SEP]
hypo: [CLS] in this picture we can see a group of people standing on the image there are some other objects. [SEP]
refe: [CLS] in this image i can see a girl sitting on the road, around her there are so many baskets with some food items and some other objects, also there are so many, also there are few people standing at the road. [SEP]
hypo: [CLS] in this picture we can see a group of people standing on the background. [SEP]
refe: [CLS] in this image at the bottom there are a group of people who are standing, and in the background there is a wall. [SEP]
hypo: [CLS] this person wearing clicked is observe a woman. in the image there are sitting on the hands. in front of her hand, we can see a table. [SEP]
refe: [CLS] in this image, i can see the woman sitting and smiling. this is a table with a book, glasses and few other things on it. i can see a person sitting and few people standing. i think these are the clothes hanging to a hanger. i can see the buildings. these look like the name boards. i think these are the current poles with the current wires. [SEP]
```

after
```
hypo: [CLS] in this image we can see a group of people. in the ground. there are standing. on the floor. [SEP]
refe: [CLS] in this image there are some persons standing at right side of this image and there are some lights are arranged at top of this image. there are some cabins as we can see at left side of this image and there is a floor at bottom of this image. [SEP]
hypo: [CLS] in this picture we can see a utens flame oven. [SEP]
refe: [CLS] in this image we can see a pie on a large spatula. on the backside we can see some fire. [SEP]
hypo: [CLS] this picture describes about are a room. in the image, there is a cake. on the background, we can see a table. [SEP]
refe: [CLS] in this image, we can see a table, on that table there is a cake and we can see a candle on the cake, there is a black color laptop on the table, in the background we can see a wall and there are some photos on the wall. [SEP]
hypo: [CLS] in this picture we can see there are placed on the floor. [SEP]
refe: [CLS] in the image we can see a t - shirt, black in color, on it there is a text. we can even see there are many flower bookies. this is a footpath, brick wall and fruits. [SEP]
hypo: [CLS] in this image there are observe a group of people standing on the floor. in the background we can see the ground. [SEP]
refe: [CLS] in this image in the foreground there are two persons one person is wearing a hat, and in the background there are group of people one person is holding a pole and flag and some boards poles, buildings, mountains. at the bottom there are some stones, and at the top there is sky. [SEP]
hypo: [CLS] in this picture we can see a vehicle. [SEP]
refe: [CLS] in this picture we can see a wheel, a gear and springs in the front, in the background there is grass. [SEP]
hypo: [CLS] in this picture we can see a person standing on the floor. in her hand. in the ground. [SEP]
refe: [CLS] here we can see a woman having a football in her hand and behind her we can see group of people standing and it is snowy [SEP]
hypo: [CLS] this image a picture are a room. on the right side of a laptop, we can see a bottle, there is a table. [SEP]
refe: [CLS] in this image there is a bed, on the bed there are the pillows and beside the bed there is a table, on the table there is a laptop and there is a fish pot, under the table there is a bottle kept on the table and there is a speaker on the floor. [SEP]
hypo: [CLS] this picture is clicked outside buildings. at the ground. in the road, there are trees. [SEP]
refe: [CLS] in this picture we can see vehicles on the road, trees, poles and in the background we can see the sky with clouds. [SEP]
hypo: [CLS] in this picture we can see a building. in the path. in the image. in the right side there are trees. on the road. [SEP]
refe: [CLS] in this picture, we can see a few buildings, poles, stairs, railing, road, a few people, and vehicles, and the sky with clouds, and we can see date and time in bottom right side of the picture. [SEP]
hypo: [CLS] this image a picture is a room. in the right side, we can see a table, there are some other objects. [SEP]
refe: [CLS] this image is taken in a store where we can see mirrors, chairs, wall, lights, ceiling, bottles, tripod stands, frames, jar, table and a cupboard. [SEP]
hypo: [CLS] in this picture, we observe a room. here we can see a chair, there is a table. [SEP]
refe: [CLS] this is a picture of a kitchen. in this picture there are chairs, table, microwave oven, lights closet, mat and many other kitchen utensils. on the left there is a refrigerator, in the refrigerator there are drinks and other food items. [SEP]
hypo: [CLS] in front a person is a man standing in the ground. in the image there are walking on the background we can see a building. [SEP]
refe: [CLS] in this image there is a person walking, plants, grass, bench, buildings, pole, antenna, trees, sky. [SEP]
hypo: [CLS] in this picture we can see a car on the ground. in the road. [SEP]
refe: [CLS] in this picture we can see car on the road and we can see grass and helmet. in the background of the image it is blurry. [SEP]
hypo: [CLS] in this image, we observe a dog on the wall, we can see the background there is a tree. [SEP]
refe: [CLS] in this image i can see an animal which is in white color. i can see the pole. in the background i can see many trees and the sky. [SEP]
hypo: [CLS] in this picture we can see a group of people standing and they are smiling. in their hands. [SEP]
refe: [CLS] in this image we can see people, rods and wall. among them two people are holding bottles and one woman wore handbag. [SEP]
hypo: [CLS] in the foreground of a building. in the image. this is walking. in the buildings. in the right side there are trees. on the road. [SEP]
refe: [CLS] in the middle of the image we can see some buildings, trees, plants and two persons walking on the road. at the top of the image there are some clouds in the sky. [SEP]
hypo: [CLS] in the foreground of a a a black goggles and she is wearing a cap, we can see the background there are trees. [SEP]
refe: [CLS] the woman in blue t - shirt and black trouser is highlighted. she wore gloves, goggles. far there are number of bare trees. snow is in white color. [SEP]
hypo: [CLS] in this image we can see a cat in the floor. on the background there is a wall. [SEP]
refe: [CLS] in this image we can see an animal on the ground. in the background, we can see the wall with ventilation. [SEP]
hypo: [CLS] in this image there are a woman who is a few people standing on the right side of the background we can see a screen. [SEP]
refe: [CLS] in this picture we can see a person wearing a black object on the head. we can see some text on the black object. there is a phone visible in the hands of a person in the bottom right. we can see screen a few people and other objects in the background. background is blurry. [SEP]
hypo: [CLS] in the bottom it is observe a group of people standing on the image we can see the background there are trees. [SEP]
refe: [CLS] in the picture i can see people among them a person is standing behind the wooden fence and the people in front of the image are wearing helmets and some other objects. in the background i can see trees, an umbrella and some other objects. [SEP]
hypo: [CLS] in this picture we can see a house in the grass. in the ground and plants and there are trees. [SEP]
refe: [CLS] in this image there is an old house and some algae on the house, around the house there are trees, plants and some dry leaves on the surface. [SEP]
hypo: [CLS] in this picture we can see a building. in the ground. in the image, there are trees. on the road. [SEP]
refe: [CLS] in this image we can see buildings, staircases, railings, sign boards, motor vehicles on the road, persons walking on the floor, electric cables, trees and sky with clouds. [SEP]
hypo: [CLS] in the foreground of a man who is a bicycle on the road. [SEP]
refe: [CLS] a man is riding a bicycle. man wears blue vest and blue denim pant with casual shoe. he has a long beard and short hair, wears spectacles. there is a wall and a pillar to left of him. from a distance, it is looking as if he is passing through a bridge. [SEP]
hypo: [CLS] in this image there is a painting. in the picture we can see a book. [SEP]
refe: [CLS] in this image we can see there is a poster with a painting. and there is a person sitting on the object. there is a pillow, umbrella, cloth, flowers and tree. [SEP]
hypo: [CLS] in this picture, we observe a person. in the image, there is a boy standing on the background we can see the ground. [SEP]
refe: [CLS] in this picture i can see a woman and a boy standing. i can see buildings, trees, hills and a blue cloudy sky. i can see grass on the ground. [SEP]
hypo: [CLS] in this picture we can see a building. [SEP]
refe: [CLS] in this image i can see buildings and windows. this image is taken may be during a day. [SEP]
hypo: [CLS] in this image we can see a butterfly in the flower. on the background there is blurred. [SEP]
refe: [CLS] in this image we can see a butterfly is sitting on white and yellow color flower. behind so many flowers are there. [SEP]
hypo: [CLS] in this image there is observe a person wearing clothes and holding an object. in front of the background we can see some objects. [SEP]
refe: [CLS] in this picture there is a woman wearing white color t - shirt with cap is holding a rod in the grill burner. behind you can see white and blue color burner. in the background there is a white cladding tiles and some plates placed on the top. [SEP]
hypo: [CLS] in this image, we observe a rat which is a tree. on the background we can see a plant. [SEP]
refe: [CLS] in this picture we can see a person ' s fingers and in the background we can see a wooden object, mice, plants with flowers and wooden sticks. [SEP]
hypo: [CLS] in the foreground of a a woman in the image there is standing on the background we can see a wall. [SEP]
refe: [CLS] in the middle of the image a woman is standing and smiling. beside her we can see a painting on a wooden board. behind her we can see a wall. [SEP]
hypo: [CLS] in the foreground of a woman wearing a girl standing and she is smiling, we can see the background, plants, there are trees. [SEP]
refe: [CLS] in this picture we can see a girl, she is standing in front of the plants, in the background we can see a room and few trees. [SEP]
hypo: [CLS] in the image, in this is a few people standing on the ground. in front of them, we can see the background there are trees. [SEP]
refe: [CLS] in this picture i can see group of people, there are baskets, those are looking like sweet potatoes, and in the background there are trees. [SEP]
hypo: [CLS] this picture describes about are wearing a man sitting in the chairs, we can see the chair. in front of him, there is a microphone. on the table. [SEP]
refe: [CLS] this image is taken indoors. in the background there is a wall with a door and there is a person standing on the floor. at the bottom of the image there is a table with a tablecloth, a name board and a few things on it. on the right side of the image a woman and a person are sitting on the chairs and a man is standing on the floor. in the middle of the image a man is sitting on the chair. on the left side of
hypo: [CLS] in this picture we can see two men sitting. in the chairs. in the chair, there are glasses. in front of them. on the table. [SEP]
refe: [CLS] in this picture there are three men who are sitting on the chair. there is a book. there is a glass. there is a spectacle on the table. there is a frame on the wall. there is a poster. there is a door and there is a device. [SEP]
hypo: [CLS] in this picture we can see a christmas tree. [SEP]
refe: [CLS] in this image in the center there is a tree and some lights and decorations, and there is black background. [SEP]
hypo: [CLS] in this image we can see a poster. [SEP]
refe: [CLS] this image consists of a poster and this is a black and white image. on the left side a person is wearing a jacket, standing and looking at the picture. on the right side, i can see some text. [SEP]
hypo: [CLS] in this picture we can see a house. on the image there is a building. [SEP]
refe: [CLS] this is the picture of a shed. in this image there is a person walking. there is a text and painting on the wall. in the foreground there are pipes. at the back there are trees and poles. at the top there is sky. [SEP]
hypo: [CLS] in this picture we can see a bottle on the table. [SEP]
refe: [CLS] this is a picture, in this picture there is a green color bottle and a wine glass on a wooden table. background of this bottle and a glass is a wall which is in white color. [SEP]
hypo: [CLS] in this image there are a two persons and a woman is a man standing on the background we can see a table. [SEP]
refe: [CLS] in this image i can see a woman sitting on chair and 2 person standing behind her. i can see a table on which i can see a jar and 2 plates. [SEP]
hypo: [CLS] this are a black and white image there is a woman sitting on the background. [SEP]
refe: [CLS] in this picture there is a man who is wearing t - shirt and short. she is holding a flower. in the back i can see the darkness. [SEP]
hypo: [CLS] in the picture, in this is a few people and smiling. in front of the image there are standing on the background we can see some other objects. [SEP]
refe: [CLS] in this image there are people. there is a table on the right side and there are glasses. there is a cloth at the top. there are trees. there is sky. [SEP]
hypo: [CLS] in this image there are observe a few people and smiling and holding an me is a man standing on the background we can see a banner. [SEP]
refe: [CLS] in this image, we can see people and some are wearing uniforms and holding boards with some text and one of them is wearing a hat. in the background, there are some objects and we can see a cardboard and there is a wall. [SEP]
hypo: [CLS] in this image men are two persons standing. in front of the background there is a microphone. [SEP]
refe: [CLS] in this image i can see on the left side a man is holding the microphone, he wore black color coat, on the right side there is another man. he wore green color coat, white color cap, behind them there is the white color cloth to an iron grill. [SEP]
hypo: [CLS] this image a man is observe a group of people standing on the backside we can see the background there are trees. [SEP]
refe: [CLS] in the middle a man is there, he wore white color cap and a garland. here a beautiful woman is standing, she wore orange color dress beside him a man is there, he wore t - shirt, goggles. there are trees at the backside of an image. on the right side there is a house. [SEP]
hypo: [CLS] in this image we can see a poster. [SEP]
refe: [CLS] this is an edited image. in this image, in the middle, we can see a mobile, in the mobile, we can see some pictures and a keyboard. on the right side, we can see some text. [SEP]
hypo: [CLS] in this picture we can see leaves. [SEP]
refe: [CLS] in this picture we can see fruits in the water. [SEP]
hypo: [CLS] in this image there are a woman who is a man standing and holding a dog. in the background we can see the ground. [SEP]
refe: [CLS] in this picture there are three persons one, two, three, one is a lady among them, a lady is holding the baby and the person who is standing at the right side of the image it seems to be walking by holding the leach of the dog and the person who is standing in the middle of the image is watching in front of the direction and the area where they were stood it seems to be greenery and there is a building at the right side of the image on
hypo: [CLS] in this picture, black observe a woman in the image there is standing on the background we can see a tree. [SEP]
refe: [CLS] this picture is an animated image. we can see a lady is standing on the left side of the image. on the right side of the image a christmas tree is there. in the background of the image we can see a wall. at the bottom of the image floor is there. [SEP]
hypo: [CLS] this picture an clicked is a train. in the track. in front of the image, we can see the background there are trees. on the road. [SEP]
refe: [CLS] in the picture we can see a street, in the street we can see a train on the track which is green in color with some yellow lines and besides, we can see some people standing on the path near to the buildings and we can also see some poles, street lights, wires to the poles and sky with a cloud. [SEP]
hypo: [CLS] in this image, we can see a few people standing on the floor. in front of them there are some objects. [SEP]
refe: [CLS] in the image few people are standing and sitting and smiling and there are some wires and there is a chair. they are in a wooden building and there is a window. through the window we can see some trees and there is a vehicle. [SEP]
hypo: [CLS] in this picture we can see a plant. [SEP]
refe: [CLS] in this image we can see flowers and buds. in the top right, we can see a rock. the background of the image is blurred. [SEP]
hypo: [CLS] in this picture we can see a sign board on the image there is a tree. [SEP]
refe: [CLS] in the center of the image we can see a sign board attached to the pole. in the background we can see the tree. [SEP]
hypo: [CLS] in the foreground of a city. in the railway track. it moving on it is a train. i can see the background there are trees. [SEP]
refe: [CLS] in this image there is a metro train under the metro train there is a house, near the house there is a road on that road there are light poles and signal poles, cars are parked to a side, in the background there is a big building. [SEP]
hypo: [CLS] in the a black and white image which is a sculpture we can see the background there are lights. [SEP]
refe: [CLS] in this image we can see sculptures and a wall. behind the sculptures we can see lamps and water. the background of the image is dark. [SEP]
hypo: [CLS] in this image, there is a man standing on the floor. in front of the stage, we can see a microphone. [SEP]
refe: [CLS] in this picture, we can see a few people, we can see the stage, and some objects on the stage like microphones with stands, speakers, and in the background we can see the screen. [SEP]
hypo: [CLS] in this image there is a two men and a man standing on the background we can see a wall. [SEP]
refe: [CLS] this is an edited picture. i can see three persons standing, there is a person holding a camera, and in the background there is a door and a board. [SEP]
hypo: [CLS] in this picture in the image there are toys, we can see the floor. [SEP]
refe: [CLS] in this picture, we see the toys which are holding some electronic goods. we see these toys are placed on the white table or thermocol. beside that, we see an objects in black color which looks like the solar panels. on either side of the panels, we see the walls, which are made up of thermocol. we even see the railing and this wall is in blue color. in the background, we see a building in white color. this
hypo: [CLS] in this image, we observe a girl is standing on the ground, we can see the background there are trees. [SEP]
refe: [CLS] in this image i can see a person standing. she is wearing green, black and white dress and black shoe. background i can see trees and wall. [SEP]
hypo: [CLS] this are a black and white picture. in the image there is a man standing on the background we can see a door. [SEP]
refe: [CLS] in this picture there is a man standing on the floor and carrying a bag and we can see glass, through glass we can see car on the road, rods, people and buildings. in the background of the image we can see a board on the wall. [SEP]
hypo: [CLS] in this picture, we can see a few people sitting in the floor. in front of them there are some objects. on the table. [SEP]
refe: [CLS] in the center of the image there are four people sitting around a dining table behind them there are three people standing. in the background there is a door and a curtain. on the left side of the room there is a bin. on the right side there is a cupboard. there are glasses, cups, spoons and plates placed on the table. [SEP]
hypo: [CLS] in this picture we can see a group of people standing on the image there are some other objects. [SEP]
refe: [CLS] in this image i can see a girl sitting on the road, around her there are so many baskets with some food items and some other objects, also there are so many, also there are few people standing at the road. [SEP]
hypo: [CLS] in this picture we can see a group of people standing on the background. [SEP]
refe: [CLS] in this image at the bottom there are a group of people who are standing, and in the background there is a wall. [SEP]
hypo: [CLS] in this image there is observe a woman sitting on the chair and smiling. in her hand. in front of the background we can see a table. [SEP]
refe: [CLS] in this image, i can see the woman sitting and smiling. this is a table with a book, glasses and few other things on it. i can see a person sitting and few people standing. i think these are the clothes hanging to a hanger. i can see the buildings. these look like the name boards. i think these are the current poles with the current wires. [SEP]

```

## Summary

Comparing the results before and after reinforcement learning, improvements are observed in all metrics except `repeat_count`. Even for `repeat_count`, the values ​​are 0.324 and 0.328, representing a deterioration of only 1.23%.
