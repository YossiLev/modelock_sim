(function(){var v;htmx.defineExtension("ws",{init:function(e){v=e;if(!htmx.createWebSocket){htmx.createWebSocket=t}if(!htmx.config.wsReconnectDelay){htmx.config.wsReconnectDelay="full-jitter"}},onEvent:function(e,t){var n=t.target||t.detail.elt;switch(e){case"htmx:beforeCleanupElement":var r=v.getInternalData(n);if(r.webSocket){r.webSocket.close()}return;case"htmx:beforeProcessNode":h(l(n,"ws-connect"),function(e){s(e)});h(l(n,"ws-send"),function(e){a(e)})}}});function i(e){return e.trim().split(/\s+/)}function r(e){var t=v.getAttributeValue(e,"hx-ws");if(t){var n=i(t);for(var r=0;r<n.length;r++){var s=n[r].split(/:(.+)/);if(s[0]==="connect"){return s[1]}}}}function s(a){if(!v.bodyContains(a)){return}var e=v.getAttributeValue(a,"ws-connect");if(e==null||e===""){var t=r(a);if(t==null){return}else{e=t}}if(e.indexOf("/")===0){var n=location.hostname+(location.port?":"+location.port:"");if(location.protocol==="https:"){e="wss://"+n+e}else if(location.protocol==="http:"){e="ws://"+n+e}}var o=c(a,function(){return htmx.createWebSocket(e)});o.addEventListener("message",function(e){if(m(a)){return}var t=e.data;if(!v.triggerEvent(a,"htmx:wsBeforeMessage",{message:t,socketWrapper:o.publicInterface})){return}v.withExtensions(a,function(e){t=e.transformResponse(t,null,a)});var n=v.makeSettleInfo(a);var r=v.makeFragment(t);if(r.children.length){var s=Array.from(r.children);for(var i=0;i<s.length;i++){v.oobSwap(v.getAttributeValue(s[i],"hx-swap-oob")||"true",s[i],n)}}v.settleImmediately(n.tasks);v.triggerEvent(a,"htmx:wsAfterMessage",{message:t,socketWrapper:o.publicInterface})});v.getInternalData(a).webSocket=o}function c(r,t){var s={socket:null,messageQueue:[],retryCount:0,events:{},addEventListener:function(e,t){if(this.socket){this.socket.addEventListener(e,t)}if(!this.events[e]){this.events[e]=[]}this.events[e].push(t)},sendImmediately:function(e,t){if(!this.socket){v.triggerErrorEvent()}if(!t||v.triggerEvent(t,"htmx:wsBeforeSend",{message:e,socketWrapper:this.publicInterface})){this.socket.send(e);t&&v.triggerEvent(t,"htmx:wsAfterSend",{message:e,socketWrapper:this.publicInterface})}},send:function(e,t){if(this.socket.readyState!==this.socket.OPEN){this.messageQueue.push({message:e,sendElt:t})}else{this.sendImmediately(e,t)}},handleQueuedMessages:function(){while(this.messageQueue.length>0){var e=this.messageQueue[0];if(this.socket.readyState===this.socket.OPEN){this.sendImmediately(e.message,e.sendElt);this.messageQueue.shift()}else{break}}},init:function(){if(this.socket&&this.socket.readyState===this.socket.OPEN){this.socket.close()}var n=t();v.triggerEvent(r,"htmx:wsConnecting",{event:{type:"connecting"}});this.socket=n;n.onopen=function(e){s.retryCount=0;v.triggerEvent(r,"htmx:wsOpen",{event:e,socketWrapper:s.publicInterface});s.handleQueuedMessages()};n.onclose=function(e){if(!m(r)&&[1006,1012,1013].indexOf(e.code)>=0){var t=f(s.retryCount);setTimeout(function(){s.retryCount+=1;s.init()},t)}v.triggerEvent(r,"htmx:wsClose",{event:e,socketWrapper:s.publicInterface})};n.onerror=function(e){v.triggerErrorEvent(r,"htmx:wsError",{error:e,socketWrapper:s});m(r)};var e=this.events;Object.keys(e).forEach(function(t){e[t].forEach(function(e){n.addEventListener(t,e)})})},close:function(){this.socket.close()}};s.init();s.publicInterface={send:s.send.bind(s),sendImmediately:s.sendImmediately.bind(s),reconnect:s.init.bind(s),queue:s.messageQueue};return s}function a(e){var t=v.getAttributeValue(e,"hx-ws");if(t&&t!=="send"){return}var n=v.getClosestMatch(e,o);u(n,e)}function o(e){return v.getInternalData(e).webSocket!=null}function u(g,d){var t=v.getInternalData(d);var e=v.getTriggerSpecs(d);e.forEach(function(e){v.addTriggerHandler(d,e,t,function(e,t){if(m(g)){return}var n=v.getInternalData(g).webSocket;var r=v.getHeaders(d,v.getTarget(d));var s=v.getInputValues(d,"post");var i=s.errors;var a=Object.assign({},s.values);var o=v.getExpressionVars(d);var c=v.mergeObjects(a,o);var u=v.filterValues(c,d);var f={parameters:u,unfilteredParameters:c,headers:r,errors:i,triggeringEvent:t,messageBody:undefined,socketWrapper:n.publicInterface};if(!v.triggerEvent(e,"htmx:wsConfigSend",f)){return}if(i&&i.length>0){v.triggerEvent(e,"htmx:validation:halted",i);return}var l=f.messageBody;if(l===undefined){var h=Object.assign({},f.parameters);if(f.headers){h.HEADERS=r}l=JSON.stringify(h)}n.send(l,e);if(t&&v.shouldCancel(t,e)){t.preventDefault()}})})}function f(e){var t=htmx.config.wsReconnectDelay;if(typeof t==="function"){return t(e)}if(t==="full-jitter"){var n=Math.min(e,6);var r=1e3*Math.pow(2,n);return r*Math.random()}logError('htmx.config.wsReconnectDelay must either be a function or the string "full-jitter"')}function m(e){if(!v.bodyContains(e)){var t=v.getInternalData(e);if(t.webSocket){t.webSocket.close();return true}return false}return false}function t(e){var t=new WebSocket(e,[]);t.binaryType=htmx.config.wsBinaryType;return t}function l(e,t){var n=[];if(v.hasAttribute(e,t)||v.hasAttribute(e,"hx-ws")){n.push(e)}e.querySelectorAll("["+t+"], [data-"+t+"], [data-hx-ws], [hx-ws]").forEach(function(e){n.push(e)});return n}function h(e,t){if(e){for(var n=0;n<e.length;n++){t(e[n])}}}})();