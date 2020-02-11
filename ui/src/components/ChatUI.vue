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
      'uttence1': '',
      'uttence2': ''
    }
    this.process(data)
  },
  methods: {
    process: async function(data){
      var chat = this.chat['chatui']
      console.log(chat)
      let input = await chat.userInput('text')
      data['uttence2'] = input.text
      console.log(data)
      let headers = {
        'Content-Type': 'application/json'
      }
      axios.post('http://localhost:5000/query', data, headers).then(res => {
        console.log(res.data)
        chat.addEntry(res.data).readable
        console.log("done")
        data['uttence1'] = res.data
        this.process(data)
      })
    }
  }
}
</script>