from argparse import ArgumentParser
import sys
import boto3
from xml.dom.minidom import parseString

import os
import sqlite3

import pdb

import time
from datetime import datetime


def delete_hit(client, hit_id, default_accept=True):
    past_time = datetime(2015, 1, 1)

    client.update_expiration_for_hit(HITId=hit_id, ExpireAt=past_time)

    assignments = client.list_assignments_for_hit(HITId=hit_id)
    for assignment in assignments['Assignments']:
        if assignment['AssignmentStatus'] == 'Submitted':
            if default_accept:
                client.approve_assignment(
                    AssignmentId=assignment['AssignmentId']
                    )
            else:
                client.reject_assignment(
                    AssignmentId=assignment['AssignmentId']
                    )

    client.delete_hit(HITId=hit_id)


def print_hit_info(client, hit_id):
    ignore_list = ['Description', 'Question', 'QualificationRequirements', 'Keywords', 'Reward']

    print("\n[getting hit information]...\n")

    hit = client.get_hit(HITId=hit_id)['HIT']
    for hit_key in hit.keys():
        if hit_key not in ignore_list:
            print("{}:{}".format(hit_key, hit[hit_key]))


def print_assignment_infos(client, hit_id):
    assignments = client.list_assignments_for_hit(HITId=hit_id)['Assignments']

    if len(assignments) > 0:
        print("\n[getting assignments]...\n")
        for assignment in assignments:
            for a_key in assignment:
                print("{}:{}".format(a_key, assignment[a_key]))
            print("")
    else:
        print("[No assignments for this hit]")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--profile_name', type=str, default=None)
    parser.add_argument('--live', action='store_true')

    # possible actions
    parser.add_argument('--action', choices=['get_info', 'delete', 'review'], default='servers')
    parser.add_argument('--get_info', action='store_true')
    parser.add_argument('--delete', action='store_true')
    #parser.add_argument('--delete', action='store_true')

    # arguments for actions
    parser.add_argument('--hit_id', default=None)
    parser.add_argument('--tag', type=str, default=None)
    #parser.add_argument('--expire_hit', action='store_true')


    args = parser.parse_args()

    pdb.set_trace()

    live = args.live
    hit_id = args.hit_id
    tag = args.tag

    environments = {
        "live": {
            "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
            "preview": "https://www.mturk.com/mturk/preview",
            "manage": "https://requester.mturk.com/mturk/manageHITs",
            #"reward": "0.00"
        },
        "sandbox": {
            "endpoint": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
            "preview": "https://workersandbox.mturk.com/mturk/preview",
            "manage": "https://requestersandbox.mturk.com/mturk/manageHITs",
            #"reward": "0.11"
        },
    }
    
    if live:
        mturk_environment = environments["live"]
        db_path = "mturk.db"
    else:
        mturk_environment = environments["sandbox"]
        db_path = "mturk_sandbox.db"

    # use profile if one was passed as an arg, otherwise
    profile_name = args.profile_name
    session = boto3.Session(profile_name=profile_name)
    client = session.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint'],
    )

    hit_ids = []
    assert os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()


    if hit_id:
        hit_ids.append(hit_id)
    elif tag:
        #c.execute('SELECT hit_id FROM hits WHERE tag=?', (tag,))
        if tag == 'None':
            c.execute('SELECT hit_id FROM hits WHERE tag IS null')
        else:
            c.execute('SELECT hit_id FROM hits WHERE tag=?', (tag,))
        #c.execute('SELECT tag FROM hits')
        hit_ids = [x[0] for x in c.fetchall()]
    else:
        hits = client.list_hits(MaxResults=100)['HITs']
        for i, hit in enumerate(hits):
            print("\n{}:{}:{}".format(i+1,hit['HITId'],hit['HITStatus']))
        sys.exit()

    #for hit_id in hit_ids:
    #    delete_hit(client, hit_id)
        #s.append(client.get_hit(HITId=hit_id))

    hits = client.list_hits(MaxResults=100)['HITs']
    pdb.set_trace()


'''
    if not hit_id:
        hits = client.list_hits(MaxResults=100)['HITs']
        hitTypes = set()
        for i, hit in enumerate(hits):
            print("\n{}:{}:{}".format(i+1,hit['HITId'],hit['HITStatus']))
            #assignments = client.list_assignments_for_hit(HITId=hit['HITId'])['Assignments']
            #pdb.set_trace()
            #for assignment in assignments:
            #    if assignment['AssignmentStatus'] != 'Approved':
            #        client.approve_assignment(
            #            AssignmentId=assignment['AssignmentId']
            #            )

            #for assignment in hit
            #past_time = datetime(2015, 1, 1)
            #if hit['HITId'] not in ['3P7QK0GJ3TM7DNUHXZGSSNJXW232ZN', '3RWB1RTQDJOOLYU0Q7RRBGUM83O8PB', '385MDVINFCG3PONKTX2DS0BWHBKWJD', '3RWO3EJELHA6AYAFRMICW67EUXHP17']:
            #    client.delete_hit(HITId=hit['HITId'])
            #client.update_expiration_for_hit(HITId=hit['HITId'], ExpireAt=past_time)
            #client.delete_hit(HITId=hit['HITId'])
            #for hit_key in hit.keys(): print("{}:{}".format(hit_key, hit[hit_key]))
            #if hit['HITStatus'] == 'Assignable':
            #    work_url = "https://workersandbox.mturk.com/mturk/preview?groupId={}".format(hit['HITTypeId'])
            #    print(work_url)
            #    hitTypes.update([work_url])

    else:
        print_hit_info(client, hit_id)

        print_assignment_infos(client, hit_id)

        if args.delete:
            client.delete_hit(HITId=hit_id)

    #pdb.set_trace()
 

def expire_hit(client, hit_id):
    import time
    pdb.set_trace()
    expire_at = datetime.datetime.fromtimestamp(time.time() + 20)
    response = client.update_expiration_for_hit(
        HITId=hit_id,
        ExpireAt=expire_at
    )
    return response


    hit = client.get_hit(HITId=hit_id)
    print 'Hit {} status: {}'.format(hit_id, hit['HIT']['HITStatus'])
    response = client.list_assignments_for_hit(
        HITId=hit_id,
        AssignmentStatuses=['Submitted', 'Approved'],
        MaxResults=10,
    )

assignments = response['Assignments']
print 'The number of submitted assignments is {}'.format(len(assignments))
for assignment in assignments:
    worker_id = assignment['WorkerId']
    assignment_id = assignment['AssignmentId']
    answer_xml = parseString(assignment['Answer'])

    # the answer is an xml document. we pull out the value of the first
    # //QuestionFormAnswers/Answer/FreeText
    answer = answer_xml.getElementsByTagName('FreeText')[0]
    # See https://stackoverflow.com/questions/317413
    only_answer = " ".join(t.nodeValue for t in answer.childNodes if t.nodeType == t.TEXT_NODE)

    print 'The Worker with ID {} submitted assignment {} and gave the answer "{}"'.format(worker_id, assignment_id, only_answer)

    # Approve the Assignment (if it hasn't already been approved)
    if assignment['AssignmentStatus'] == 'Submitted':
        decision = raw_input('Approve assignment? [Y|N]')
        if decision == 'Y':
            print 'Approving Assignment {}'.format(assignment_id)
            client.approve_assignment(
                AssignmentId=assignment_id,
                RequesterFeedback='good',
                OverrideRejection=False,
            )
'''