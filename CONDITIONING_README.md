# Conditioning explanations
Here we will list out all the conditionings the model accepts as well as a short description and some tips for optimal use. For conditionings with a learned unconditional, they can be set to that to allow the model to infer an appropriate setting.
### espeak
- **Type:** `EspeakPhonemeConditioner`
- **Description:**  
  Responsible for cleaning, phonemicizing, tokenizing, and embedding the text provided to the model. This is the text pre-processing pipeline. If you would like to change how a word is pronounced or enter raw phonemes you can do that here.

  Supported by transformer and hybrid models.
---
### speaker
- **Type:** `PassthroughConditioner`
- **Attributes:**
  - **cond_dim:** `128`
  - **uncond_type:** `learned`
  - **projection:** `linear`
- **Description:**  
  An embedded representation of the speakers voice. We use [these](https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding) speaker embedding models. It can capture a surprising amount of detail from the reference clip and supports arbitrary length input. Try to input clean reference clips containing only speech. It can be valid to concatenate multiple clean samples from the same speaker into one long sample and may lead to better cloning. If the speaker clip is very long, it is advisable to cut out long speech-free background music segments if they exist. If the reference clip is yielding noisy outputs with denoising enabled we recommend doing source separation before cloning.

  Supported by transformer and hybrid models.
---
### emotion
- **Type:** `FourierConditioner`
- **Attributes:**
  - **input_dim:** `8`
  - **uncond_type:** `learned`
- **Description:**  
  Encodes emotion in an 8D vector. Included emotions are Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral in that order. This vector tends to be entangled with various other conditioning inputs. More notably, it's entangled with text based on the text sentiment (eg. Angry texts will be more effectively conditioned to be angry, but if you try to make it sound sad it will be a lot less effective). It's also entangled with pitch standard deviation since larger values there tend to correlate to more emotional utterances. It's also heavily correlated with VQScore and DNSMOS as these conditionings favor neutral speech. It's also possible to do a form of "negative prompting" by doing CFG where the unconditional branch is set to a highly neutral emotion vector instead of the true unconditional value, doing this will exaggerate the emotions as it pushes the model away from being neutral.

  Supported by transformer and hybrid models.
---
### fmax
- **Type:** `FourierConditioner`
- **Attributes:**
  - **min_val:** `0`
  - **max_val:** `24000`
  - **uncond_type:** `learned`
- **Description:**  
  Specifies the max frequency of the audio. For best results select 22050 or 24000 as these correspond to 44.1 and 48KHz audio respectively. They should not be any different in terms of actual max frequency since the model's sampling rate is 44.1KHz but they represent different slices of data which lead to slightly different voicing. Selecting a lower value generally produces lower-quality results both in terms of acoustics and voicing.

  For voice cloning it is recommended to use 22050.

  Supported by transformer and hybrid models.
---
### pitch_std
- **Type:** `FourierConditioner`
- **Attributes:**
  - **min_val:** `0`
  - **max_val:** `400`
  - **uncond_type:** `learned`
- **Description:**  
  Specifies the standard deviation of the pitch of the output audio. Wider variations of pitch tend to be more correlated with expressive speech. Good values are from 20-45 for normal speech and 60-150 for expressive speech. Higher than that generally tend to be crazier samples.

  Supported by transformer and hybrid models.
---
### speaking_rate
- **Type:** `FourierConditioner`
- **Attributes:**
  - **min_val:** `0`
  - **max_val:** `40`
  - **uncond_type:** `learned`
- **Description:**  
  Specifies the number of phonemes to be read per second. When entering a long text, it is advisable to adjust the speaking rate such that the number of phonemes is readable within the generation length. For example, if your generation length is 10 seconds, and your input is 300 phonemes, you would want either 30 phonemes per second (which is very very fast) or to generate a longer sample. The model's maximum is 30 seconds. Please note that unrealistic speaking rates can be OOD for the model and create undesirable effects, so at the 30-second limit, it can be better to cut the text short and do multiple generations than to feed the model the entire prompt and have an unrealistically low speaking rate.
  
  Supported by transformer and hybrid models.
---
### language_id
- **Type:** `IntegerConditioner`
- **Attributes:**
  - **min_val:** `-1`
  - **max_val:** `126`
  - **uncond_type:** `learned`
- **Description:**  
  Indicates which language the output should be in. A mapping for these values can be found in the [conditioning section](https://github.com/Zyphra/Zonos/blob/3807c8e04bd4beaadb9502b3df1ffa4b0350e3f7/zonos/conditioning.py#L308C1-L376C21) of Zonos.

  Supported by transformer and hybrid models.
---
### vqscore_8
- **Type:** `FourierConditioner`
- **Attributes:**
  - **input_dim:** `8`
  - **min_val:** `0.5`
  - **max_val:** `0.8`
  - **uncond_type:** `learned`
- **Description:**  
  Encodes the desired [VQScore](https://github.com/JasonSWFu/VQscore) value for the output audio. VQScore is an unsupervised speech quality (cleanliness) estimation method that we found has superior generalization and reduced biases compared to supervised methods like DNSMOS. A good value for our model is 0.78 for high-quality speech. The eight dimensions correspond to consecutive 1/8th chunks of the audio. (eg. for an 8-second output, the first dimension represents the quality of the first second only). For inference, we generally set all 8 dimensions to the same value. This has an unfortunately strong correlation with expressiveness, so for expressive speech, we recommend setting it to unconditional.

  Only applicable for the hybrid model.
---
### ctc_loss
- **Type:** `FourierConditioner`
- **Attributes:**
  - **min_val:** `-1.0`
  - **max_val:** `1000`
  - **uncond_type:** `learned`
- **Description:**  
  Encodes loss values from a [CTC](https://en.wikipedia.org/wiki/Connectionist_temporal_classification) (Connectionist Temporal Classification) setup, this indicates how well the training-time transcription matched with the audio according to a CTC model. For inference always use low values (eg. 0.0 or 1.0)

  Only applicable for the hybrid model.
---
### dnsmos_ovrl
- **Type:** `FourierConditioner`
- **Attributes:**
  - **min_val:** `1`
  - **max_val:** `5`
  - **uncond_type:** `learned`
- **Description:**  
  A [MOS](https://arxiv.org/abs/2110.01763) score for the output audio. This is similar to VQScore and tends to have a stronger entanglement with emotions. It additionally has a strong entanglement with languages. Set to 4.0 for very clean and neutral English speech, else we recommend setting it to unconditional.

  Only applicable for the hybrid model.
---
### speaker_noised
- **Type:** `IntegerConditioner`
- **Attributes:**
  - **min_val:** `0`
  - **max_val:** `1`
  - **uncond_type:** `learned`
- **Description:**  
  Indicates if the speaker embedding is noisy or not. If checked this lets the model clean (denoise) the input speaker embedding. When this is set to True, VQScore and DNSMOS will have a lot more power to clean the speaker embedding, so for very noisy input samples we recommend setting this to True and specifying a high VQScore value. If your speaker cloning outputs sound echo-y or do weird things, setting this to True will help.

  Only applicable for the hybrid model.
