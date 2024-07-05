# Install

Install [this branch](https://github.com/butia-bots/butia_vision_msgs/tree/feature/gpsr-recognition) of butia_vision_msgs. Then run the following commands on the jetson, and make sure the pre-installed version of pytorch, numpy and other libraries from JetPack SDK is kept frozen and not updated during the install process.

```sh
pip install inference supervision transformers accelerate peft
```