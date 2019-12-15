<template>
  <div id="app">
    <b-card>
      <b-card-header header-bg-variant="info">
        <h2>チャットボット</h2>
        何か話しかけて見てね
      </b-card-header>
      <b-card-body body-bg-variant='dark'>
        <ChatUI/>
      </b-card-body>
    </b-card>
  </div>
</template>

<script>
import ChatUI from './components/ChatUI.vue'

export default {
  name: 'app',
  components: {
    ChatUI
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
</style>
