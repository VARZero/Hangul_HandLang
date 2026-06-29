import unicodedata

JA = 1
MO = 2
NO_UMSO = 0

jaum = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
moum = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

CHOSEONG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSEONG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSEONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def checkJAMO(UMSO):
    if UMSO in jaum:
        return JA
    elif UMSO in moum:
        return MO
    else:
        return NO_UMSO

def assemble(cho, jung, jong=None):
    try:
        cho_idx = CHOSEONG.index(cho)
        jung_idx = JUNGSEONG.index(jung)
        jong_idx = JONGSEONG.index(jong) if jong else 0
        return chr(0xAC00 + (cho_idx * 21 + jung_idx) * 28 + jong_idx)
    except ValueError:
        res = cho + jung
        if jong:
            res += jong
        return res

class hangelAlphabetQueue:
    def __init__(self):
        self.q = list()
        self.popUMSO = ""

    def __add__(self, newChar): # 음소 추가
        self.q.append(newChar)
        return self

    def popUMJEOAL(self): # 음절단위 출력
        if not self.q:
            return ""
        
        # 1. 초성 (자음이어야 함)
        if checkJAMO(self.q[0]) != JA:
            return self.q.pop(0)
        
        cho = self.q.pop(0)
        
        # 2. 중성 (모음이어야 함)
        if not self.q or checkJAMO(self.q[0]) != MO:
            return cho
            
        jung = self.q.pop(0)
        
        # 3. 종성 (자음이어야 하고, 그 뒤에 바로 모음이 오지 않아야 함)
        jong = None
        if self.q and checkJAMO(self.q[0]) == JA:
            if len(self.q) > 1 and checkJAMO(self.q[1]) == MO:
                # 다음 음소가 모음이면 현재 자음은 다음 음절의 초성이 되어야 하므로 종성으로 취하지 않음
                pass
            else:
                jong = self.q.pop(0)
                
        return assemble(cho, jung, jong)

    def gatherUMJEOAL(self):
        result = ""
        while self.q:
            result += self.popUMJEOAL()
        return result

if __name__ == '__main__':
    q = hangelAlphabetQueue()
    while 1:
        inStr = input()
        if inStr != 'e':
            q + inStr
        else: break
    
    print("결과:", q.gatherUMJEOAL())
