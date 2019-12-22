from entity_linker_front import EntityLinker
import logging

def run():
    test_question = 'which films star by Forest Whitaker and are directed by Mark Rydell'
    el = EntityLinker()
    entities = el.link(test_question)

    for e in entities:
        print (e.entity.id, e.score)


if __name__ == '__main__':
    run()