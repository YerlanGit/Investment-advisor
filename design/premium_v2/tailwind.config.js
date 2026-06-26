module.exports = {
  content: ['./design/**/*.jsx','./design/**/*.html'],
  theme: { extend: {
    fontFamily: { sans:['Manrope','system-ui','sans-serif'], mono:['"JetBrains Mono"','ui-monospace','monospace'] },
    colors: {
      cream:{50:'#fbf8f1',100:'#f6f3ea',200:'#efe9d8'},
      ink:{950:'#100f0e',900:'#1c1b1a',800:'#2a2825',700:'#3a3833',600:'#55524c',500:'#6b6862',400:'#9a958c',300:'#c4bfb5',200:'#e2ddd1'},
      gold:{300:'#fbe48a',400:'#f5d04e',500:'#eac233',600:'#caa01a',700:'#9a7a10'},
      sage:{500:'#7a9a78',600:'#5d7c5c'}, rust:{500:'#c47358',600:'#a85a40'},
    },
    borderRadius:{'4xl':'32px','5xl':'40px'},
    boxShadow:{
      'card':'0 1px 0 rgba(255,255,255,0.7) inset, 0 24px 48px -28px rgba(60,50,30,0.18), 0 1px 2px rgba(60,50,30,0.05)',
      'card-lg':'0 1px 0 rgba(255,255,255,0.7) inset, 0 36px 72px -36px rgba(60,50,30,0.28), 0 1px 2px rgba(60,50,30,0.06)',
      'dark':'0 1px 0 rgba(255,255,255,0.06) inset, 0 24px 48px -24px rgba(0,0,0,0.45)',
    },
  } },
}
