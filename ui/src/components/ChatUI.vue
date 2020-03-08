<template>
  <vue-chatui ref="chatui"></vue-chatui>
</template>

<script>
import axios from 'axios'

export default{
  name: 'ChatUI',
  data: function(){
    return {
      chat: this.$refs
    }
  },
  mounted: function(){
    let data = {
      'history': [],
      'uttence': ''
    }
    this.process(data)
  },
  methods: {
    process: async function(data){
      var chat = this.chat['chatui']
      console.log(chat)
      let input = await chat.userInput('text')
      data['uttence'] = input.text
      console.log(data)
      let headers = {
        'Content-Type': 'application/json',
        "Access-Control-Allow-Origin": "*"
      }
      // axios.post('https://m2vajnoqb5.execute-api.ap-northeast-1.amazonaws.com/dev/talk2', data, headers).then(res => {
      axios.post('http://localhost:5000/query', data, headers).then(res => {
        console.log(res.data)
        chat.addEntry(res.data.response).readable
        console.log("done")
        data['history'] = res.data.history
        this.process(data)
      })
    }
  }
}
</script>