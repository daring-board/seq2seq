import Vue from 'vue'
import App from './App.vue'
import BootstrapVue from 'bootstrap-vue'
import VueChatui from '@daniel-ordonez/vue-chatui'

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
import "@daniel-ordonez/vue-chatui/dist/VueChatui.umd.js"
import "@daniel-ordonez/vue-chatui/dist/VueChatui.css"

Vue.config.productionTip = false
Vue.use(BootstrapVue)
Vue.component('vue-chatui', VueChatui)

new Vue({
  render: h => h(App),
}).$mount('#app')
